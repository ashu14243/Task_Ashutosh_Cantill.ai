"""Orchestrates classification of included lines into the company's tree.

Evidence sources, combined as probability vectors over leaves:
  1. zero-shot lexical match against the tree (always on; the only source for a brand-new company)
  2. supervised text model + item/supplier priors from labelled examples (if the config has labels)
  3. optional LLM on the highest-spend distinct lines (if an API key / local endpoint is configured)
  4. self-training: a text model fitted on the confident predictions, to generalise to lines that
     share no words with the tree (part numbers, abbreviations)
  5. propagation: a supplier's / item's confident lines inform its uncertain ones
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .models import TextModel, ZeroShotLexical
from .taxonomy import Taxonomy

log = logging.getLogger(__name__)

FIELDS = ["description", "gl_description", "supplier_name", "extra_text"]


def _unit_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    key = df[FIELDS].fillna("").astype(str).apply(lambda s: s.str.strip().str.upper())
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(key))
    units = pd.DataFrame(list(uniques), columns=FIELDS)
    return units, codes


def _model_text(units: pd.DataFrame) -> list[str]:
    return (units["description"] + " | " + units["gl_description"] + " | " + units["supplier_name"] + " | " +
            units["extra_text"]).tolist()


def _key_prior(keys: pd.Series, leaf: pd.Series, n: int, min_n: int = 1) -> dict:
    t = pd.DataFrame({"k": keys, "y": leaf}).dropna()
    t = t[t.k.astype(str).ne("")]
    out = {}
    for k, g in t.groupby("k")["y"]:
        if len(g) < min_n:
            continue
        v = np.bincount(g.astype(int), minlength=n).astype(float)
        out[k] = (v / v.sum(), len(g))
    return out


def _topdown(P: np.ndarray, tax: Taxonomy) -> tuple[np.ndarray, np.ndarray]:
    """Pick the most probable L1 (summing its leaves), then the best L2 inside it, and so on.
    Scoring is per level, so getting the upper levels right matters as much as the leaf.
    Confidence = probability mass of the chosen leaf's whole path, averaged over levels."""
    n, L = P.shape
    chosen = np.full(n, -1)
    conf = np.zeros(n)
    cand = np.ones((n, L), dtype=bool)
    level_mass = []
    for d in range(tax.max_depth):
        keys = [lf[:d + 1] if len(lf) > d else lf for lf in tax.leaves]
        uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        M = np.zeros((L, len(uniq)))
        for j, k in enumerate(keys):
            M[j, uniq[k]] = 1.0
        mass = (P * cand) @ M                      # probability of each node at this depth among candidates
        best = mass.argmax(axis=1)
        level_mass.append(mass[np.arange(n), best])
        node_of_leaf = np.array([uniq[k] for k in keys])
        cand &= node_of_leaf[None, :] == best[:, None]
    chosen = np.where(cand, P, -1).argmax(axis=1)
    conf = np.mean(level_mass, axis=0)
    return chosen, conf


def classify(df: pd.DataFrame, tax: Taxonomy, cfg: dict, labels: pd.DataFrame | None,
             include_mask: pd.Series) -> pd.DataFrame:
    cc = cfg["classification"]
    fw = cc.get("text_fields") or {}
    L = tax.n
    rows = df[include_mask].copy()
    out = pd.DataFrame(index=df.index, data={"leaf": -1, "confidence": 0.0, "class_source": ""})
    if rows.empty:
        return out
    units, codes = _unit_frame(rows)
    nu = len(units)
    unit_spend = np.bincount(codes, weights=rows["spend_rc"].abs().fillna(0).to_numpy(), minlength=nu)
    unit_count = np.bincount(codes, minlength=nu).astype(float)
    log.info("classification: %d included lines, %d distinct line texts, %d leaves", len(rows), nu, L)

    # ---- 1. zero-shot
    zs = ZeroShotLexical(tax, temperature=float(cc.get("zero_shot_temperature", 0.06)),
                         alpha=float(cc.get("zero_shot_length_norm", 0.75)))
    field_texts = [(units[f].tolist(), float(fw.get(f, 1.0))) for f in FIELDS if float(fw.get(f, 0)) > 0]
    P_zs, evid = zs.predict(field_texts)
    P = P_zs.copy()
    src = np.where(evid > 0, "zero_shot", "none").astype(object)
    texts = _model_text(units)

    # ---- 2. supervised from labels
    lab_leaf = None
    if labels is not None and cc.get("use_labels", True) and "_leaf" in labels.columns:
        lab_leaf = df[["record_id"]].join(labels[["_leaf"]], on="record_id")["_leaf"]
        lab_rows = lab_leaf.notna() & include_mask
        if lab_rows.sum() >= 20:
            lr = df[lab_rows]
            lu, lcodes = _unit_frame(lr)
            y = lab_leaf[lab_rows].astype(int).to_numpy()
            pairs = pd.DataFrame({"u": lcodes, "y": y}).value_counts().reset_index(name="w")
            ltexts = _model_text(lu)
            tm = TextModel(L, C=float(cc.get("supervised_C", 4.0)))
            tm.fit([ltexts[u] for u in pairs.u], pairs.y.to_numpy(), np.sqrt(pairs.w.to_numpy(dtype=float)))
            P_sup, sim = tm.predict_proba(texts)
            a = np.clip(sim, 0, 1) ** 2 * float(cc.get("supervised_weight", 0.9))
            a[sim > 0.999] = 0.97
            P = a[:, None] * P_sup + (1 - a[:, None]) * P
            src = np.where(a > 0.5, "supervised", src)
            log.info("supervised model: %d labelled lines, %d distinct texts; mean similarity of new lines %.2f",
                     int(lab_rows.sum()), len(lu), float(np.mean(sim)))

    # ---- 3. optional LLM
    llm_cfg = cc.get("llm") or {}
    try:
        from .llm import llm_classify
        P_llm, got = llm_classify(units, unit_spend, tax, llm_cfg)
        if got.any():
            w = float(llm_cfg.get("weight", 0.6))
            P[got] = (1 - w) * P[got] + w * P_llm[got]
            src[got] = "llm"
    except Exception as e:  # never let an optional API break an unattended run
        log.warning("LLM step skipped: %s", e)

    # ---- 4. self-training on confident predictions
    if cc.get("self_training", True) and nu >= 50:
        conf = P.max(axis=1)
        y = P.argmax(axis=1)
        ok = (conf >= np.quantile(conf, 0.4)) & (src != "none")
        if ok.sum() >= 30 and len(np.unique(y[ok])) >= 2:
            st = TextModel(L, C=2.0)
            st.fit([texts[i] for i in np.where(ok)[0]], y[ok], np.sqrt(unit_count[ok]) * conf[ok])
            P_st, sim_st = st.predict_proba(texts)
            weak = (src == "none") | (conf < 0.35)
            b = np.where(weak, 0.6, 0.25)
            P = (1 - b[:, None]) * P + b[:, None] * P_st
            src = np.where((src == "none") & (P_st.max(axis=1) > 0.2), "self_training", src)
            log.info("self-training on %d confident distinct texts", int(ok.sum()))

    # ---- 5. row level: priors and propagation
    Pr = P[codes]
    srcr = src[codes].astype(object)
    spend = rows["spend_rc"].abs().fillna(0).to_numpy()
    if lab_leaf is not None:
        lab_known = lab_leaf[include_mask].notna()
        for key, wmax in (("item_key", 0.85), ("supplier_key", 0.35)):
            pri = _key_prior(rows.loc[lab_known, key], lab_leaf[include_mask][lab_known], L)
            if not pri:
                continue
            for i, k in enumerate(rows[key].to_numpy()):
                if k in pri:
                    dist, n = pri[k]
                    purity = dist.max()
                    w = wmax * purity * min(1.0, n / 3.0)
                    Pr[i] = (1 - w) * Pr[i] + w * dist
                    if key == "item_key" and w > 0.5:
                        srcr[i] = "item_prior"
    # propagation within item and supplier, using confident rows, weighted by spend
    conf_r = Pr.max(axis=1)
    for key, strength in (("item_key", 0.5), ("supplier_key", 0.35)):
        k = rows[key].to_numpy()
        wts = conf_r * (np.log1p(spend) + 1.0)
        agg = pd.DataFrame(Pr * wts[:, None]).groupby(k).sum()
        cnt = pd.Series(1, index=range(len(k))).groupby(k).sum()
        agg = agg[cnt.reindex(agg.index) >= 2]
        if agg.empty:
            continue
        D = agg.div(agg.sum(axis=1).replace(0, 1), axis=0)
        has = pd.Series(k).isin(D.index).to_numpy()
        weak = has & (conf_r < 0.5)
        if weak.any():
            Dk = D.loc[k[weak]].to_numpy()
            Pr[weak] = (1 - strength) * Pr[weak] + strength * Dk
            srcr[weak & (srcr == "none")] = f"{key}_propagation"
    Pr = Pr / Pr.sum(axis=1, keepdims=True)
    if cc.get("decode", "flat") == "topdown":
        leaf, conf = _topdown(Pr, tax)
    else:
        leaf, conf = Pr.argmax(axis=1), Pr.max(axis=1)

    # labelled rows keep their label
    if lab_leaf is not None:
        lk = lab_leaf[include_mask]
        m = lk.notna().to_numpy()
        leaf[m] = lk[m].astype(int).to_numpy()
        conf[m] = 1.0
        srcr[m] = "label"
    out.loc[rows.index, "leaf"] = leaf
    out.loc[rows.index, "confidence"] = np.round(np.clip(conf, 0, 1), 4)
    out.loc[rows.index, "class_source"] = srcr
    return out
