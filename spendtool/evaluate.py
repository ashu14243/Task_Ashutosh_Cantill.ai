"""Measure the tool against a config's labels.

zero-label : the run never sees the labels (exactly the situation of a new company)
holdout    : 20% of labelled records are hidden, the rest are used as training labels
             (the situation of A2/B2: new records from a company whose labels we have)
Reports category accuracy per level (by count and by spend) and B-cubed precision/recall/F1 of the
supplier and item keys against the golden keys.
"""
from __future__ import annotations

import copy
import json
import logging

import numpy as np
import pandas as pd

from .classify.taxonomy import load_taxonomy
from .pipeline import load_labels, run

log = logging.getLogger(__name__)


def bcubed(pred: pd.Series, gold: pd.Series) -> dict:
    """B-cubed precision/recall/F1 of a clustering (pred) against gold clusters."""
    t = pd.DataFrame({"p": pred.astype(str).to_numpy(), "g": gold.astype(str).to_numpy()})
    n_pg = t.groupby(["p", "g"]).size()
    n_p = t.groupby("p").size()
    n_g = t.groupby("g").size()
    j = n_pg.reset_index(name="n")
    j["np"] = j.p.map(n_p)
    j["ng"] = j.g.map(n_g)
    prec = float((j.n * j.n / j.np).sum() / len(t))
    rec = float((j.n * j.n / j.ng).sum() / len(t))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "pred_clusters": int(len(n_p)), "gold_clusters": int(len(n_g))}


def _accuracy(out: pd.DataFrame, gold_paths: pd.DataFrame, rows: pd.Index, depth: int) -> dict:
    o = out.loc[rows]
    g = gold_paths.loc[rows]
    w = o["spend"].abs().to_numpy()
    res = {}
    for k in range(1, depth + 1):
        pred = o[[f"l{i}" for i in range(1, k + 1)]].astype(str).agg(" > ".join, axis=1).str.lower()
        true = g[[f"l{i}" for i in range(1, k + 1)]].astype(str).agg(" > ".join, axis=1).str.lower()
        ok = (pred == true).to_numpy()
        res[f"L{k}"] = {"by_count": round(float(ok.mean()), 4),
                        "by_spend": round(float((ok * w).sum() / max(w.sum(), 1e-9)), 4)}
    return res


def evaluate(cfg: dict, with_labels: bool = False, holdout: float = 0.2, seed: int = 7) -> dict:
    tax = load_taxonomy(cfg)
    labels = load_labels(cfg, tax)
    if labels is None or "_leaf" not in labels.columns:
        raise SystemExit("evaluate needs a labels block with category_columns in the config")
    gold = labels.copy()
    paths = np.full((len(gold), 4), "", dtype=object)
    for i, lf in enumerate(gold["_leaf"]):
        if pd.notna(lf):
            p = tax.leaves[int(lf)]
            paths[i, :len(p)] = p
    gold_paths = pd.DataFrame(paths, index=gold.index, columns=["l1", "l2", "l3", "l4"])
    report = {}

    # ---------------------------------------------------------------- zero-label
    c0 = copy.deepcopy(cfg)
    c0["output"]["diagnostics"] = False
    res = run(c0, use_labels=False, output=str(_tmp(cfg, "zero_label")))
    out = res["frame"].set_index("record_id")
    rows = gold.index[gold["_leaf"].notna()].intersection(out.index[out["include"] == 1])
    report["zero_label"] = {"rows_scored": int(len(rows)), "accuracy": _accuracy(out, gold_paths, rows, tax.max_depth)}
    for key in ("supplier_key", "item_key"):
        if key in gold.columns:
            r = gold.index[gold[key].astype(str).ne("")].intersection(out.index)
            report["zero_label"][key] = bcubed(out.loc[r, key], gold.loc[r, key])
    if "exclusion_reason" in gold.columns:
        r = gold.index.intersection(out.index)
        g_inc = gold.loc[r, "exclusion_reason"].eq("")
        p_inc = out.loc[r, "include"].eq(1)
        report["zero_label"]["include_agreement"] = round(float((g_inc == p_inc).mean()), 4)
        report["zero_label"]["non_procurement_recall"] = round(float((~p_inc[~g_inc]).mean()), 4) if (~g_inc).any() else None
    report["zero_label"]["runtime_seconds"] = res["summary"]["runtime_seconds"]
    log.info("ZERO-LABEL: %s", json.dumps(report["zero_label"], indent=1))

    # ---------------------------------------------------------------- holdout
    if with_labels:
        rng = np.random.default_rng(seed)
        lab_ids = gold.index[gold["_leaf"].notna()]
        hide = pd.Index(rng.choice(lab_ids, size=int(len(lab_ids) * holdout), replace=False))
        import spendtool.pipeline as pl
        orig = pl.load_labels

        def hidden_loader(cfg_, tax_):
            lab = orig(cfg_, tax_)
            return lab.drop(index=hide, errors="ignore")
        pl.load_labels = hidden_loader
        try:
            c1 = copy.deepcopy(cfg)
            c1["output"]["diagnostics"] = False
            res1 = pl.run(c1, use_labels=True, output=str(_tmp(cfg, "holdout")))
        finally:
            pl.load_labels = orig
        out1 = res1["frame"].set_index("record_id")
        rows = hide.intersection(out1.index[out1["include"] == 1])
        report["holdout"] = {"rows_scored": int(len(rows)), "accuracy": _accuracy(out1, gold_paths, rows, tax.max_depth)}
        log.info("HOLDOUT (%.0f%% labels hidden): %s", holdout * 100, json.dumps(report["holdout"], indent=1))
    path = _tmp(cfg, "evaluation").with_suffix(".json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def _tmp(cfg, tag):
    from pathlib import Path
    d = Path(cfg["_config_path"]).parent.parent / "outputs" / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{cfg['company']}_{tag}.csv"
