"""End-to-end run: config -> one output row per input record."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .classify.engine import classify
from .classify.taxonomy import Taxonomy, load_taxonomy
from .config import resolve_path
from .dedup import assign_item_keys, assign_supplier_keys
from .ingest import _read_table, load_sources
from .matching import cross_match, prepare, within_source_duplicates
from .spend_rules import INCLUDE, map_status, non_procurement
from .valuation import compute_net, convert_currency, parse_dates

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
                  "l1", "l2", "l3", "l4", "confidence"]
TRAIN_PREFIX = "__train__"


def training_config(cfg: dict) -> dict | None:
    """Config describing the labelled training files, or None when the run already uses them.

    `labels.training_files: {source_name: path}` names the files the labels belong to. If a run
    points a source at a different file (e.g. --input erp=A2_erp.csv), those training files are
    loaded alongside."""
    lc = cfg.get("labels") or {}
    tf = lc.get("training_files") or {}
    if not tf:
        return None
    same = all(resolve_path(cfg, tf.get(s["name"], s["file"])).resolve() == resolve_path(cfg, s["file"]).resolve()
               for s in cfg["sources"])
    if same:
        return None
    import copy
    t = copy.deepcopy(cfg)
    srcs = []
    for s in t["sources"]:
        if s["name"] in tf:
            s["file"] = tf[s["name"]]
            srcs.append(s)
    if not srcs:
        return None
    for s in srcs:
        if not resolve_path(cfg, s["file"]).exists():
            log.warning("training file %s not found; continuing without training rows", s["file"])
            return None
    t["sources"] = srcs
    return t


def load_labels(cfg: dict, tax: Taxonomy) -> pd.DataFrame | None:
    lc = cfg.get("labels")
    if not lc or not lc.get("file"):
        return None
    path = resolve_path(cfg, lc["file"])
    if not path.exists():
        log.warning("labels file %s not found; running without labels", path)
        return None
    raw = _read_table(path, lc.get("read", {}))
    fmt = lc.get("id_format", "{id}")
    out = pd.DataFrame({"record_id": [fmt.format(id=v) for v in raw[lc.get("id_column", "record_id")]]})
    cats = lc.get("category_columns") or []
    if cats:
        vals = raw[cats].fillna("").astype(str).to_numpy()
        cache = {}
        leaves = []
        for row in vals:
            k = tuple(row)
            if k not in cache:
                cache[k] = tax.find_leaf(*k) if any(x.strip() for x in k) else None
            leaves.append(cache[k])
        out["_leaf"] = pd.array(leaves, dtype="Int64")
        miss = sum(1 for k, v in cache.items() if v is None and any(x.strip() for x in k))
        if miss:
            log.warning("labels: %d label paths are not in the taxonomy and are ignored", miss)
    for src, dst in (("supplier_key_column", "supplier_key"), ("item_key_column", "item_key")):
        if lc.get(src) and lc[src] in raw.columns:
            out[dst] = raw[lc[src]].fillna("").astype(str)
    if lc.get("include_column") and lc["include_column"] in raw.columns:
        inc = raw[lc["include_column"]].astype(str).str.strip()
        reason = raw[lc["exclusion_column"]].fillna("").astype(str) if lc.get("exclusion_column") else ""
        out["exclusion_reason"] = np.where(inc.isin(["1", "1.0", "True", "true"]), "", reason)
        # excluded rows carry no category
        if "_leaf" in out.columns:
            out.loc[out["exclusion_reason"].ne(""), "_leaf"] = pd.NA
    out = out.drop_duplicates("record_id").set_index("record_id")
    log.info("labels: %d labelled records from %s", len(out), path)
    return out


def run(cfg: dict, use_labels: bool = True, output: str | None = None) -> dict:
    t0 = time.time()
    tax = load_taxonomy(cfg)
    log.info("taxonomy: %d leaves, depth %d", tax.n, tax.max_depth)
    labels = load_labels(cfg, tax) if use_labels else None

    df = load_sources(cfg)
    df["_train"] = False
    # When the run's input files are not the files the labels describe (e.g. the blind-test A2/B2
    # files), the labelled training files are loaded too. Their rows inform supplier/item keys and
    # the classifier, but never enter matching, spend or the output.
    train_cfg = training_config(cfg) if labels is not None else None
    if train_cfg is not None:
        tdf = load_sources(train_cfg)
        tdf["record_id"] = TRAIN_PREFIX + tdf["record_id"]
        tdf["_train"] = True
        tdf["_uid"] = tdf["_uid"] + len(df)
        # labels now only attach to the training copies, never to run rows that reuse an ID
        labels = labels.set_axis(TRAIN_PREFIX + labels.index.astype(str))
        log.info("training rows loaded from the labelled files: %d", len(tdf))
        df = pd.concat([df, tdf], ignore_index=True)
    df = parse_dates(df, cfg)
    df = compute_net(df, cfg)
    df = convert_currency(df, cfg)

    # ---- status
    status_res = pd.Series(INCLUDE, index=df.index, dtype=object)
    has_status = df["status_present"].astype(bool)
    if has_status.any():
        status_res[has_status] = map_status(df.loc[has_status, "status"], cfg)
    reason = status_res.where(status_res != INCLUDE, "")
    rule = pd.Series(np.where(reason.ne(""), "status:" + df["status"].astype(str), ""), index=df.index, dtype=object)

    # ---- non-procurement
    np_hit, np_why = non_procurement(df, cfg, labels)
    zero_rule = cfg["non_procurement"].get("zero_amount", "include")
    if str(zero_rule).upper() == "NON_PROCUREMENT":
        z = df["spend_rc"].fillna(0).eq(0)
        np_hit |= z
        np_why[z & np_why.eq("")] = "zero_amount"
    m = reason.eq("") & np_hit
    reason[m] = "NON_PROCUREMENT"
    rule[m] = np_why[m]

    # ---- suppliers and items
    df["supplier_key"] = assign_supplier_keys(df, cfg, labels)
    df["item_key"], df["item_how"] = assign_item_keys(df, cfg, df["supplier_key"], labels)

    # ---- count once
    df = prepare(df)
    is_train = df["_train"].astype(bool)
    eligible = reason.eq("") & df["spend_rc"].notna() & ~is_train
    dup = pd.Series(False, index=df.index)
    dup_to = pd.Series("", index=df.index, dtype=object)
    dup_how = pd.Series("", index=df.index, dtype=object)
    mcfg = cfg.get("matching")
    if mcfg:
        d, to, how = cross_match(df[~is_train], eligible[~is_train], mcfg)
        d, to, how = (d.reindex(df.index, fill_value=False), to.reindex(df.index, fill_value=""),
                      how.reindex(df.index, fill_value=""))
        dup |= d
        dup_to[d], dup_how[d] = to[d], how[d]
        # a matched secondary line is the same purchase: give it the primary line's keys
        rid_to_idx = pd.Series(df.index, index=df["record_id"]).groupby(level=0).first()
        src_idx = rid_to_idx.reindex(to[d]).to_numpy()
        df.loc[d, "item_key"] = df.loc[src_idx, "item_key"].to_numpy()
        df.loc[d, "supplier_key"] = df.loc[src_idx, "supplier_key"].to_numpy()
    for wrule in (cfg.get("duplicates") or {}).get("within_source") or []:
        d, to = within_source_duplicates(df, eligible, wrule, already=dup)
        d &= ~dup
        dup |= d
        dup_to[d], dup_how[d] = to[d], "exported_twice"
    m = reason.eq("") & dup
    reason[m] = "DUPLICATE"
    rule[m] = "dup:" + dup_how[m] + "->" + dup_to[m]

    include = reason.eq("")
    # ---- classify (training rows ride along so their labels teach the model)
    cls = classify(df, tax, cfg, labels, include)

    # ---- drop training rows from everything that is reported
    keep = ~is_train
    df, reason, rule, include, cls = df[keep], reason[keep], rule[keep], include[keep], cls[keep]

    # ---- output
    out = pd.DataFrame({"record_id": df["record_id"], "include": include.astype(int), "exclusion_reason": reason,
                        "supplier_key": df["supplier_key"], "item_key": df["item_key"]})
    out["spend"] = np.where(include, df["spend_rc"].fillna(0.0), 0.0).round(2)
    lv = np.full((len(df), 4), "", dtype=object)
    leaves = cls["leaf"].to_numpy()
    for i in np.where(leaves >= 0)[0]:
        p = tax.leaves[leaves[i]]
        lv[i, :len(p)] = p[:4]
    for k in range(4):
        out[f"l{k + 1}"] = lv[:, k]
    conf = cls["confidence"].to_numpy(dtype=float)
    # confidence of an exclusion decision
    exc_conf = np.select([reason.eq("CANCELLED") | reason.eq("NOT_APPROVED"),
                          reason.eq("NON_PROCUREMENT") & rule.str.startswith(("default", "config", "learned")),
                          reason.eq("NON_PROCUREMENT"),
                          rule.str.contains("po\\+amount"), reason.eq("DUPLICATE")],
                         [1.0, 0.9, 0.7, 0.95, 0.8], default=0.0)
    out["confidence"] = np.where(include, conf, exc_conf).round(4)

    out_path = Path(output) if output else _default_output(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[OUTPUT_COLUMNS].to_csv(out_path, index=False)

    # diagnostics (local only, not part of the deliverable)
    if cfg["output"].get("diagnostics", True):
        diag = pd.DataFrame({
            "record_id": df["record_id"], "source": df["source"], "rule": rule, "status_raw": df["status"],
            "amount_raw": df["amount"], "amount_type": df["amount_type"], "tax_rate_used": df["tax_rate_used"],
            "net_doc": df["net_doc"], "currency": df["currency_norm"], "fx_rate": df["fx_rate_used"],
            "fx_how": df["fx_how"], "spend_rc": df["spend_rc"], "date_raw": df["date"],
            "date": df["date_parsed"].dt.strftime("%Y-%m-%d"), "date_flag": df["date_flag"],
            "fx_date_flag": df["fx_date_flag"], "amount_flag": df["amount_flag"], "fx_flag": df["fx_flag"],
            "item_how": df["item_how"], "class_source": cls["class_source"], "description": df["description"]})
        diag.to_csv(out_path.with_suffix(".diagnostics.csv"), index=False)

    summary = summarize(out, df, reason, rule, cfg, time.time() - t0, out_path)
    summary["validation_problems"] = validate_output(out, df, tax, cfg)
    with open(out_path.with_suffix(".summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return {"output": str(out_path), "summary": summary, "frame": out, "work": df, "classes": cls, "taxonomy": tax}


def _default_output(cfg: dict) -> Path:
    oc = cfg["output"]
    name = oc.get("file") or f"{cfg.get('candidate_name', 'candidate')}__{cfg['company']}.csv"
    d = Path(oc.get("dir", "outputs"))
    if not d.is_absolute():
        d = Path(cfg["_config_path"]).parent.parent / d
    return d / name


def summarize(out, df, reason, rule, cfg, secs, path) -> dict:
    s = {
        "company": cfg["company"], "output": str(path), "rows": int(len(out)),
        "rows_by_source": df["source"].value_counts().to_dict(),
        "included": int(out["include"].sum()),
        "excluded_by_reason": reason[reason.ne("")].value_counts().to_dict(),
        "total_spend": round(float(out["spend"].sum()), 2),
        "spend_by_source": out.groupby(df["source"])["spend"].sum().round(2).to_dict(),
        "reporting_currency": cfg["reporting_currency"],
        "date_flags": df["date_flag"].value_counts().to_dict(),
        "fx_how": df["fx_how"].value_counts().to_dict(),
        "unparseable_amounts": int(df["amount_flag"].ne("").sum()),
        "no_fx_rate": int(df["fx_flag"].ne("").sum()),
        "suppliers": int(out["supplier_key"].nunique()), "items": int(out["item_key"].nunique()),
        "mean_confidence_included": round(float(out.loc[out["include"] == 1, "confidence"].mean() or 0), 3),
        "top_rules": rule[rule.ne("")].str.split("->").str[0].value_counts().head(15).to_dict(),
        "runtime_seconds": round(secs, 1),
    }
    log.info("summary: %s", json.dumps({k: s[k] for k in ("rows", "included", "excluded_by_reason", "total_spend",
                                                            "suppliers", "items", "runtime_seconds")}, default=str))
    return s


def validate_output(out: pd.DataFrame, df: pd.DataFrame, tax: Taxonomy, cfg: dict) -> list[str]:
    problems = []
    if len(out) != len(df):
        problems.append("row count differs from input")
    if list(out.columns[:11]) != OUTPUT_COLUMNS:
        problems.append("columns differ from template")
    exc = out["include"] == 0
    if (out.loc[exc, "spend"] != 0).any():
        problems.append("excluded rows with non-zero spend")
    if out.loc[exc, "exclusion_reason"].eq("").any():
        problems.append("excluded rows without reason")
    if out.loc[~exc, "exclusion_reason"].ne("").any():
        problems.append("included rows with a reason")
    valid = set(tax.leaves)
    paths = out.loc[~exc, ["l1", "l2", "l3", "l4"]].apply(lambda r: tuple(x for x in r if x), axis=1)
    bad = ~paths.isin(valid)
    if bad.any():
        problems.append(f"{int(bad.sum())} included rows with a path not in the taxonomy")
    if out["spend"].isna().any():
        problems.append("NaN spend")
    if ((out["confidence"] < 0) | (out["confidence"] > 1)).any():
        problems.append("confidence outside 0..1")
    for p in problems:
        log.error("VALIDATION: %s", p)
    if not problems:
        log.info("validation passed: %d rows, all included rows carry a valid tree path", len(out))
    return problems
