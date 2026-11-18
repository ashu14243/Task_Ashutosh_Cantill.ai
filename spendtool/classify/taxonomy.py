"""Read a category tree at run time. No category names live in code.

Supported layouts (auto-detected, or set `taxonomy.format`):
  * levels:        one row per leaf, one column per level (category_l1, category_l2, ...).
                   Blank cells below a level mean the leaf sits higher in the tree.
  * parent_child:  columns id, parent_id, name (set taxonomy.id_column / parent_column / name_column).
Optional `taxonomy.description_columns` add free text (definitions, examples) to a leaf.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ..config import resolve_path
from ..ingest import _read_table


@dataclass
class Taxonomy:
    leaves: list[tuple[str, ...]]                 # full paths, one per classifiable leaf
    max_depth: int
    descriptions: dict[tuple[str, ...], str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.leaves)

    def index_of(self, path) -> int | None:
        path = tuple(p for p in path if p)
        try:
            return self.leaves.index(path)
        except ValueError:
            return None

    def find_leaf(self, *names) -> int | None:
        """Map a labelled path (possibly with case/space differences) to a leaf index."""
        key = tuple(_canon(n) for n in names if n and str(n).strip())
        for i, lf in enumerate(self.leaves):
            if tuple(_canon(x) for x in lf) == key:
                return i
        # labelled deeper/shallower than tree: match on longest common prefix uniquely
        for k in range(len(key), 0, -1):
            hits = [i for i, lf in enumerate(self.leaves) if tuple(_canon(x) for x in lf[:k]) == key[:k]]
            if len(hits) == 1:
                return hits[0]
        return None


def _canon(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _level_columns(df: pd.DataFrame, cfg_cols) -> list[str]:
    if cfg_cols:
        return list(cfg_cols)
    cols = list(df.columns)
    pat = re.compile(r"(l|lvl|level|cat(egory)?[_\s]*l(evel)?)[_\s\-]*(\d)$|^(level|l)[_\s]*(\d)", re.I)
    found = []
    for c in cols:
        m = pat.search(c.strip())
        if m:
            d = next(g for g in m.groups()[::-1] if g and g.isdigit())
            found.append((int(d), c))
    if found:
        return [c for _, c in sorted(found)]
    return [c for c in cols if not re.search(r"desc|example|definition|note|comment|code|id$", c, re.I)]


def load_taxonomy(cfg: dict) -> Taxonomy:
    tc = cfg["taxonomy"]
    df = _read_table(resolve_path(cfg, tc["file"]), tc.get("read", {}))
    df = df.fillna("")
    max_levels = int(tc.get("max_levels", 4))
    fmt = tc.get("format") or ("parent_child" if {"parent_id", "name"} <= set(c.lower() for c in df.columns) else "levels")
    desc_cols = tc.get("description_columns") or [c for c in df.columns if re.search(r"desc|example|definition|keyword", c, re.I)]
    leaves: list[tuple[str, ...]] = []
    descriptions: dict = {}
    if fmt == "parent_child":
        cols = {c.lower(): c for c in df.columns}
        idc = tc.get("id_column") or cols.get("id")
        pc = tc.get("parent_column") or cols.get("parent_id")
        nc = tc.get("name_column") or cols.get("name")
        rows = {str(r[idc]).strip(): (str(r[pc]).strip(), str(r[nc]).strip(), r) for _, r in df.iterrows()}
        children = {}
        for i, (p, _, _) in rows.items():
            children.setdefault(p, []).append(i)

        def path(i):
            out, seen = [], set()
            while i in rows and i not in seen:
                seen.add(i)
                p, n, _ = rows[i]
                out.append(n)
                i = p
            return tuple(reversed(out))
        for i in rows:
            if i not in children:
                lf = path(i)[:max_levels]
                if lf and lf not in leaves:
                    leaves.append(lf)
                    descriptions[lf] = " ".join(str(rows[i][2][c]) for c in desc_cols if c in df.columns)
    else:
        lvl = _level_columns(df, tc.get("level_columns"))[:max_levels]
        for _, r in df.iterrows():
            p = [str(r[c]).strip() for c in lvl]
            # trim trailing blanks; a blank in the middle ends the path
            q = []
            for x in p:
                if not x:
                    break
                q.append(x)
            lf = tuple(q)
            if lf and lf not in leaves:
                leaves.append(lf)
                d = " ".join(str(r[c]) for c in desc_cols if c in df.columns and str(r[c]).strip())
                if d:
                    descriptions[lf] = d
        # a path that is a strict prefix of another path is an internal node, not a leaf
        leaves = [lf for lf in leaves if not any(o != lf and o[:len(lf)] == lf for o in leaves)]
    if not leaves:
        raise ValueError("taxonomy has no categories")
    depth = max(len(l) for l in leaves)
    return Taxonomy(leaves=leaves, max_depth=depth, descriptions=descriptions)
