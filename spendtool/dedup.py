"""Supplier and item de-duplication.

Both use the same pattern: every row gets a *base label* (a normalised name, an ID, a code
or a compact description); base labels are merged with a union-find using exact rules,
fuzzy string similarity and (optionally) labelled examples; each final cluster gets one
stable key.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .normalize import norm_text, supplier_core

log = logging.getLogger(__name__)


class UnionFind:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if str(ra) < str(rb):
                self.p[rb] = ra
            else:
                self.p[ra] = rb
        return ra != rb


def stable_key(prefix: str, s: str) -> str:
    return f"{prefix}-{hashlib.md5(s.encode('utf-8')).hexdigest()[:10]}"


def _fuzzy_pairs(strings: list[str], threshold: float, scorer=fuzz.ratio, block_chars: int = 0):
    """Yield (i, j) index pairs of strings whose similarity >= threshold."""
    n = len(strings)
    if n < 2:
        return
    groups: dict[str, list[int]] = defaultdict(list)
    if n <= 6000:
        groups[""] = list(range(n))
    else:  # block on the first characters to keep memory bounded on very large inputs
        for i, s in enumerate(strings):
            groups[s[:max(block_chars, 1)]].append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        sub = [strings[i] for i in idx]
        m = process.cdist(sub, sub, scorer=scorer, score_cutoff=threshold, workers=-1, dtype=np.uint8)
        ii, jj = np.nonzero(np.triu(m, k=1))
        for a, b in zip(ii, jj):
            yield idx[a], idx[b]


# =========================================================================== suppliers
def assign_supplier_keys(df: pd.DataFrame, cfg: dict, labels: pd.DataFrame | None = None) -> pd.Series:
    sc = cfg["suppliers"]
    thr = float(sc.get("fuzzy_threshold", 92))
    min_len = int(sc.get("min_len_for_fuzzy", 5))
    cores = df["supplier_name"].map(supplier_core)
    sid = df["supplier_id"].str.strip().str.upper()
    base = np.where(cores.ne(""), "N:" + cores, np.where(sid.ne(""), "ID:" + sid, "UNKNOWN"))
    base = pd.Series(base, index=df.index)
    uf = UnionFind()
    golden: dict[str, Counter] = defaultdict(Counter)   # base label -> Counter(golden key)

    # 1) labelled examples: rows sharing a golden supplier key are one supplier
    if labels is not None and "supplier_key" in labels.columns:
        lab = df[["record_id"]].join(labels[["supplier_key"]], on="record_id")
        has = lab["supplier_key"].fillna("").ne("")
        for b, g in zip(base[has], lab.loc[has, "supplier_key"]):
            golden[b][g] += 1
        first = {}
        for b, g in zip(base[has], lab.loc[has, "supplier_key"]):
            if g in first:
                uf.union(first[g], b)
            else:
                first[g] = b
                uf.find(b)
    # 2) a supplier ID shared by a few names links those names
    if sc.get("link_by_supplier_id", True):
        tmp = pd.DataFrame({"sid": sid, "base": base})
        tmp = tmp[tmp.sid.ne("") & tmp.base.str.startswith("N:")]
        max_names = int(sc.get("max_names_per_id", 3))
        for s, grp in tmp.groupby("sid")["base"]:
            u = grp.unique()
            if len(u) <= max_names:  # an ID used by many names is a one-time/catch-all vendor: skip
                for x in u:
                    uf.union("ID:" + s, x)
    # 3) fuzzy on normalised names (blocked); never merge two different golden suppliers
    names = sorted({b for b in base.unique() if b.startswith("N:") and len(b) - 2 >= min_len})
    strs = [b[2:] for b in names]

    def gold_of(label):
        root = uf.find(label)
        return cluster_gold.get(root)

    cluster_gold = {}
    for b, cnt in golden.items():
        cluster_gold.setdefault(uf.find(b), cnt.most_common(1)[0][0])
    merged = 0
    for i, j in _fuzzy_pairs(strs, thr, scorer=fuzz.ratio, block_chars=2):
        a, b = names[i], names[j]
        ga, gb = gold_of(a), gold_of(b)
        if ga and gb and ga != gb:
            continue
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb:
            uf.union(a, b)
            merged += 1
            g = ga or gb
            if g:
                cluster_gold[uf.find(a)] = g
    # final keys
    roots = base.map(uf.find)
    # representative = most frequent base label in the cluster
    counts = Counter(zip(roots, base))
    rep = {}
    for (r, b), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][1])):
        rep.setdefault(r, b)
    gold_by_root = defaultdict(Counter)
    for b, cnt in golden.items():
        gold_by_root[uf.find(b)].update(cnt)
    keymap = {}
    for r in roots.unique():
        if gold_by_root.get(r):
            keymap[r] = gold_by_root[r].most_common(1)[0][0]
        elif r == "UNKNOWN":
            keymap[r] = "S-UNKNOWN"
        else:
            keymap[r] = stable_key("S", rep[r])
    log.info("suppliers: %d base labels -> %d keys (%d fuzzy merges)", base.nunique(), len(set(keymap.values())), merged)
    return roots.map(keymap)


# =========================================================================== items
GENERIC_WORDS = {
    "PRODUCT", "PRODUCTS", "PART", "PARTS", "ITEM", "ITEMS", "MATERIAL", "MATERIALS", "SERVICE", "SERVICES",
    "GOODS", "MISC", "MISCELLANEOUS", "VARIOUS", "SUNDRY", "SUNDRIES", "NO", "OBJECT", "NA", "N", "A", "NONE",
    "OTHER", "OTHERS", "GENERAL", "INVOICE", "LINE", "CHARGE", "CHARGES", "ORDER", "PO", "SEE", "ATTACHED",
    "AS", "PER", "QUOTE", "TBC", "TBD", "UNKNOWN", "DESCRIPTION", "NIL", "NULL", "X", "XX", "XXX", "FREE", "TEXT",
    "ARTIKEL", "ARTICULO", "PRODUIT", "PRODOTTO", "PRODUTO", "DIVERSOS", "DIVERS", "VARIOS", "SERVICIOS", "SERVICIO",
    "MATERIAL", "MATERIALE", "MATERIEL", "WARE", "WAREN", "LEISTUNG", "SONSTIGES", "SUPPLY", "SUPPLIES", "COST", "COSTS",
}


def is_uninformative(text_norm: str) -> bool:
    toks = [t for t in text_norm.split() if not t.isdigit()]
    if not toks:
        return True
    return all(t in GENERIC_WORDS for t in toks)


def item_text_key(desc: str) -> str:
    n = norm_text(desc)
    return re.sub(r"[^A-Z0-9]", "", n)


def assign_item_keys(df: pd.DataFrame, cfg: dict, supplier_keys: pd.Series,
                     labels: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """Return (item_key, how)."""
    ic = cfg["items"]
    code = df["item_code"].str.strip().str.upper()
    desc_norm = df["description"].map(norm_text)
    tkey = desc_norm.str.replace(r"[^A-Z0-9]", "", regex=True)
    uninf = desc_norm.map(is_uninformative)

    # generic codes: one code used for many unrelated descriptions (catch-all material)
    generic_codes = set()
    has_code = code.ne("")
    if has_code.any():
        g = pd.DataFrame({"code": code[has_code], "t": tkey[has_code]}).groupby("code")["t"]
        nd, nr = g.nunique(), g.size()
        share = nr / max(len(df), 1)
        generic_codes = set(nd[(nd >= int(ic.get("generic_code_min_distinct_texts", 25))) &
                               (share >= float(ic.get("generic_code_max_share", 0.05)))].index)
        if generic_codes:
            log.info("items: treating %d catch-all codes as no code: %s", len(generic_codes), sorted(generic_codes)[:5])
    use_code = has_code & ~code.isin(generic_codes)

    base = pd.Series("", index=df.index, dtype=object)
    base[use_code] = "C:" + code[use_code]
    txt = ~use_code & ~uninf
    base[txt] = "T:" + tkey[txt]
    un = ~use_code & uninf
    base[un] = "U:" + supplier_keys[un].astype(str) + ":" + tkey[un]

    uf = UnionFind()
    golden = defaultdict(Counter)
    if labels is not None and "item_key" in labels.columns:
        lab = df[["record_id"]].join(labels[["item_key"]], on="record_id")["item_key"].fillna("")
        has = lab.ne("")
        for b, g in zip(base[has], lab[has]):
            golden[b][g] += 1
        first = {}
        for b, g in zip(base[has], lab[has]):
            if g in first:
                uf.union(first[g], b)
            else:
                first[g] = b
                uf.find(b)

    # text-only rows whose description matches a coded catalogue entry inherit that code. This is a
    # one-way assignment (not a union) so two codes can never be chained together through text.
    how = pd.Series(np.where(use_code, "code", np.where(txt, "text", "unspecified")), index=df.index)
    text_to_code: dict[str, str] = {}
    digits = lambda t: "".join(re.findall(r"\d+", t))  # noqa: E731  numbers must agree exactly
    if ic.get("match_unlabelled_to_catalogue", True) and use_code.any() and txt.any():
        cat = pd.DataFrame({"t": tkey[use_code], "b": base[use_code]})
        cat = cat[cat.t.str.len() >= 4]
        dominant = cat.groupby("t")["b"].agg(lambda s: s.value_counts().index[0])
        by_sig: dict[str, list[str]] = defaultdict(list)
        for t in dominant.index:
            by_sig[digits(t)].append(t)
        q = sorted(set(tkey[txt]))
        thr = float(ic.get("catalogue_threshold", 90))
        for t in q:
            if t in dominant.index:
                text_to_code["T:" + t] = dominant[t]
                continue
            cands = by_sig.get(digits(t))
            if len(t) < 6 or not cands:
                continue
            r = process.extractOne(t, cands, scorer=fuzz.ratio, score_cutoff=thr)
            if r:
                text_to_code["T:" + t] = dominant[r[0]]
        mapped = base.isin(text_to_code.keys()) & txt
        how[mapped] = "text->code"

    # fuzzy merge of remaining text-only labels (OCR variants with identical numbers)
    tlabels = sorted({b for b in base[txt].unique() if len(b) >= 8 and b not in text_to_code})
    thr = float(ic.get("fuzzy_threshold", 95))
    groups: dict[str, list[str]] = defaultdict(list)
    for b in tlabels:
        groups[digits(b[2:])].append(b)
    for sig, members in groups.items():
        if len(members) < 2:
            continue
        for i, j in _fuzzy_pairs([b[2:] for b in members], thr, scorer=fuzz.ratio, block_chars=3):
            uf.union(members[i], members[j])

    eff = base.map(lambda b: text_to_code.get(b, b))
    roots = eff.map(uf.find)
    gold_by_root = defaultdict(Counter)
    for b, cnt in golden.items():
        gold_by_root[uf.find(b)].update(cnt)
    keymap = {}
    for r in roots.unique():
        keymap[r] = gold_by_root[r].most_common(1)[0][0] if gold_by_root.get(r) else stable_key("I", r)
    log.info("items: %d base labels -> %d keys", base.nunique(), len(set(keymap.values())))
    return roots.map(keymap), how
