"""Count every purchase once.

1. Cross-system matching (`matching:` in the config): secondary-system lines (e.g. an AP tool)
   are matched to primary-system lines (e.g. ERP goods receipts). A matched secondary line is
   excluded as DUPLICATE; the primary line is counted.
     a. same PO (+ PO line when both have it), net amount within tolerance, best quantity/text/date
     b. no usable PO: same supplier, amount within tolerance, quantity, text similarity, date window
     c. split invoices: 2-4 lines of one invoice whose amounts sum to one primary line
   Each primary line absorbs at most one secondary line in (a)/(b): a second identical invoice
   with a *different* invoice number is treated as separate spend.
2. Within-source duplicates (`duplicates.within_source`): the same invoice line exported twice
   (same supplier, invoice number, amount; compatible PO line; similar text) is counted once.
"""
from __future__ import annotations

import itertools
import logging
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

log = logging.getLogger(__name__)


def _digits_tail(s: str) -> str:
    m = re.search(r"(\d+)\D*$", s or "")
    return m.group(1).lstrip("0") if m else (s or "").strip().upper()


def _norm_po(s: str) -> str:
    d = re.sub(r"\D", "", s or "")
    return d.lstrip("0")


def _norm_line(s: str) -> str:
    try:
        return str(int(float(s))) if s not in ("", None) else ""
    except ValueError:
        return str(s).strip()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    from .normalize import norm_text
    df["_po"] = df["po_number"].map(_norm_po)
    df["_pol"] = df["po_line"].map(_norm_line)
    df["_inv"] = df["invoice_number"].map(_digits_tail)
    df["_qty"] = pd.to_numeric(df["quantity"].str.replace(",", "", regex=False), errors="coerce")
    df["_txt"] = df["description"].map(norm_text).str.replace(" ", "", regex=False)
    return df


def _tol(a, cfg):
    return np.maximum(float(cfg.get("amount_abs_tolerance", 1.0)), np.abs(a) * float(cfg.get("amount_rel_tolerance", 0.002)))


def _text_sim(a: pd.Series, b: pd.Series) -> np.ndarray:
    return np.array([fuzz.ratio(x, y) / 100.0 if x and y else 0.5 for x, y in zip(a, b)])


def _greedy(pairs: pd.DataFrame, used_p: set, used_s: set) -> list[tuple[int, int, float]]:
    out = []
    for s, p, sc in pairs.sort_values(["score", "s", "p"], ascending=[False, True, True])[["s", "p", "score"]].itertuples(index=False):
        if s in used_s or p in used_p:
            continue
        used_s.add(s)
        used_p.add(p)
        out.append((s, p, sc))
    return out


def cross_match(df: pd.DataFrame, eligible: pd.Series, mcfg: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (dup_mask, matched_to record_id, how) for secondary rows."""
    prim = df[(df["source"] == mcfg["primary"])]
    sec = df[(df["source"] == mcfg["secondary"]) & eligible]
    amt_col = "spend_rc"
    dup = pd.Series(False, index=df.index)
    to = pd.Series("", index=df.index, dtype=object)
    how = pd.Series("", index=df.index, dtype=object)
    used_p: set = set()
    used_s: set = set()
    window = float(mcfg.get("date_window_days", 120))

    def score_pairs(pr: pd.DataFrame, require_text: float | None) -> pd.DataFrame:
        if pr.empty:
            return pr
        a_s, a_p = pr["amt_s"].to_numpy(float), pr["amt_p"].to_numpy(float)
        amt_ok = np.abs(a_s - a_p) <= _tol(a_p, mcfg)
        q_s, q_p = pr["qty_s"].to_numpy(float), pr["qty_p"].to_numpy(float)
        q_known = ~np.isnan(q_s) & ~np.isnan(q_p)
        qty_ok = q_known & (np.abs(q_s - q_p) < 1e-6)
        l_s, l_p = pr["pol_s"].to_numpy(object), pr["pol_p"].to_numpy(object)
        l_known = (l_s != "") & (l_p != "")
        l_eq = l_known & (l_s == l_p)
        sim = _text_sim(pr["txt_s"], pr["txt_p"])
        d_s, d_p = pr["d_s"], pr["d_p"]
        dd = (d_s - d_p).abs().dt.days.to_numpy(dtype=float)
        date_ok = np.isnan(dd) | (dd <= window)
        ok = amt_ok & ~(l_known & ~l_eq) & ~(q_known & ~qty_ok & (sim < 0.9))
        if require_text is not None:
            ok &= (sim >= require_text) & date_ok
        sc = 10 * l_eq + 3 * qty_ok + 2 * sim - np.nan_to_num(dd, nan=180.0) / 365.0 \
            - np.abs(a_s - a_p) / np.maximum(np.abs(a_p), 1.0)
        pr = pr.assign(score=sc)
        return pr[ok]

    def make_pairs(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
        L = pd.DataFrame({"s": left.index, "k": left[on].to_numpy(), "amt_s": left[amt_col].to_numpy(),
                          "qty_s": left["_qty"].to_numpy(), "pol_s": left["_pol"].to_numpy(),
                          "txt_s": left["_txt"].to_numpy(), "d_s": left["date_parsed"].to_numpy()})
        R = pd.DataFrame({"p": right.index, "k": right[on].to_numpy(), "amt_p": right[amt_col].to_numpy(),
                          "qty_p": right["_qty"].to_numpy(), "pol_p": right["_pol"].to_numpy(),
                          "txt_p": right["_txt"].to_numpy(), "d_p": right["date_parsed"].to_numpy()})
        L, R = L[L.k != ""], R[R.k != ""]
        out = L.merge(R, on="k")
        out["d_s"] = pd.to_datetime(out["d_s"])
        out["d_p"] = pd.to_datetime(out["d_p"])
        return out

    # (a) PO-based
    pairs = score_pairs(make_pairs(sec, prim, "_po"), None)
    for s, p, _ in _greedy(pairs, used_p, used_s):
        dup[s], to[s], how[s] = True, df.at[p, "record_id"], "po+amount"
    n_a = int(dup.sum())

    # (b) supplier + amount + text (no PO, or a PO the primary system doesn't know)
    prim_pos = set(prim["_po"]) - {""}
    rest = sec[~sec.index.isin(used_s) & ~sec["_po"].isin(prim_pos)]
    free_p = prim[~prim.index.isin(used_p)]
    cand = []
    for sk, grp in rest.groupby("supplier_key"):
        pg = free_p[free_p["supplier_key"] == sk]
        if pg.empty:
            continue
        pa = pg[amt_col].to_numpy(float)
        order = np.argsort(pa)
        pa_sorted = pa[order]
        for s, a in zip(grp.index, grp[amt_col].to_numpy(float)):
            if np.isnan(a):
                continue
            t = float(_tol(a, mcfg))
            lo, hi = np.searchsorted(pa_sorted, a - t), np.searchsorted(pa_sorted, a + t, side="right")
            for k in order[lo:hi]:
                cand.append((s, pg.index[k]))
    if cand:
        c = pd.DataFrame(cand, columns=["s", "p"])
        c["k"] = "x"
        S, P = df.loc[c.s], df.loc[c.p]
        pr = pd.DataFrame({"s": c.s.to_numpy(), "p": c.p.to_numpy(), "amt_s": S[amt_col].to_numpy(),
                           "amt_p": P[amt_col].to_numpy(), "qty_s": S["_qty"].to_numpy(), "qty_p": P["_qty"].to_numpy(),
                           "pol_s": S["_pol"].to_numpy(), "pol_p": P["_pol"].to_numpy(), "txt_s": S["_txt"].to_numpy(),
                           "txt_p": P["_txt"].to_numpy(), "d_s": pd.to_datetime(S["date_parsed"].to_numpy()),
                           "d_p": pd.to_datetime(P["date_parsed"].to_numpy())})
        pr = score_pairs(pr, float(mcfg.get("text_min_similarity_no_po", 0.6)))
        for s, p, _ in _greedy(pr, used_p, used_s):
            dup[s], to[s], how[s] = True, df.at[p, "record_id"], "supplier+amount+text"
    n_b = int(dup.sum()) - n_a

    # (c) split invoices: several unmatched lines of one invoice summing to one primary line
    max_lines, max_sub = int(mcfg.get("split_max_lines", 12)), int(mcfg.get("split_max_subset", 4))
    rest = sec[~sec.index.isin(used_s)]
    prim_by_po = {k: g for k, g in prim.groupby("_po") if k}
    prim_by_sup = {k: g for k, g in prim.groupby("supplier_key")}
    n_c = 0
    for (sk, inv), grp in rest.groupby(["supplier_key", "_inv"]):
        if len(grp) < 2 or len(grp) > max_lines:
            continue
        pos = set(grp["_po"]) - {""}
        if len(pos) == 1:
            cands = prim_by_po.get(next(iter(pos)))
        elif not pos:
            cands = prim_by_sup.get(sk)
            if cands is not None and len(cands) > 2000:
                cands = None
        else:
            cands = None
        if cands is None or cands.empty:
            continue
        pa = cands[amt_col].to_numpy(float)
        idx = list(grp.index)
        amts = grp[amt_col].to_numpy(float)
        taken: set = set()
        for r in range(min(max_sub, len(idx)), 1, -1):
            for combo in itertools.combinations(range(len(idx)), r):
                if any(idx[i] in taken for i in combo):
                    continue
                tot = amts[list(combo)].sum()
                hit = np.where(np.abs(pa - tot) <= _tol(tot, mcfg))[0]
                if len(hit):
                    p = cands.index[hit[0]]
                    for i in combo:
                        s = idx[i]
                        taken.add(s)
                        used_s.add(s)
                        dup[s], to[s], how[s] = True, df.at[p, "record_id"], f"split({r})"
                        n_c += 1
    log.info("cross-system matching: %d by PO+amount, %d by supplier+amount+text, %d split-invoice lines; "
             "%d of %d eligible %s lines are duplicates of %s", n_a, n_b, n_c, int(dup.sum()), len(sec),
             mcfg["secondary"], mcfg["primary"])
    return dup, to, how


def within_source_duplicates(df: pd.DataFrame, eligible: pd.Series, rule: dict,
                             already: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Mark repeated exports of the same line. `already` = rows already counted elsewhere
    (e.g. matched to ERP); such a row is preferred as the 'original' of its group."""
    src = rule.get("source")
    keys = rule.get("keys", ["supplier_key", "_inv", "_amt2"])
    sim_min = float(rule.get("text_min_similarity", 0.75))
    distinguish = rule.get("distinguish_by", ["_pol", "invoice_line"])
    sub = df[eligible & ((df["source"] == src) if src else True)].copy()
    sub["_amt2"] = sub["spend_rc"].round(2)
    for k in keys:
        if k not in sub.columns:
            raise KeyError(f"duplicates.within_source key {k} unknown")
    sub = sub[(sub[keys].astype(str) != "").all(axis=1)]
    dup = pd.Series(False, index=df.index)
    to = pd.Series("", index=df.index, dtype=object)
    sub["_pref"] = (~already.loc[sub.index]).astype(int)
    for _, g in sub.groupby(keys, sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values(["_pref", "_uid"])
        kept: list = []
        for i, r in g.iterrows():
            orig = None
            for k in kept:
                rk = df.loc[k]
                if any(r[c] != "" and rk[c] != "" and r[c] != rk[c] for c in distinguish if c in df.columns):
                    continue
                if r["_txt"] and rk["_txt"] and fuzz.ratio(r["_txt"], rk["_txt"]) / 100.0 < sim_min:
                    continue
                orig = k
                break
            if orig is None:
                kept.append(i)
            else:
                dup[i], to[i] = True, df.at[orig, "record_id"]
    log.info("within-source duplicates (%s): %d rows", src or "all", int(dup.sum()))
    return dup, to
