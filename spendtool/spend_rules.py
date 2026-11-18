"""Decide what counts as spend: status and non-procurement rules.

Reason codes: CANCELLED, NOT_APPROVED, NON_PROCUREMENT, DUPLICATE (duplicates are set in matching.py).
Precedence when several apply: CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE.
"""
from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

INCLUDE = "include"
REASONS = ["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"]

# ordered: first match wins. Patterns are matched against the lower-cased status text.
DEFAULT_STATUS_PATTERNS = [
    (r"\b(not|un|non)[\s\-]?(approved|paid|authori[sz]ed)\b|\bunapproved\b", "NOT_APPROVED"),
    (r"cancel|void|revers|delet|withdraw|annul|storn", "CANCELLED"),
    (r"reject|declin|disput", "NOT_APPROVED"),
    (r"\bapproved\b|\bpaid\b|\bauthori[sz]ed\b|\bgenehmigt\b|\bbezahlt\b|\baprobad|\bpagad|\bpay[eé]", INCLUDE),
    (r"hold|block|submit|pend|draft|park|await|review|query|queried|\bopen\b|\bnew\b|wait|incomplete|"
     r"error|fail|escalat|routing|in[\s\-]?progress|\bdue\b|approval|unpaid|overdue", "NOT_APPROVED"),
    (r"posted|complet|closed|clear|settl|releas|accept|process|done|final|booked|matched|valid|\bok\b|"
     r"\bpay", INCLUDE),
]

# Generic, industry-independent non-procurement patterns. (field group, regex)
DEFAULT_NON_PROCUREMENT = [   # (name, field group, regex)
    # intercompany
    ("intercompany", "text", r"\binter[\s\-]?compan|\bintra[\s\-]?group\b|\bi/c\b|\bic\s*payable|\bdue\s+(to|from)\s+(affiliate|related|group)|"
             r"\brelated[\s\-]part(y|ies)\b|\bintercompany\b"),
    ("intercompany_gl", "gl", r"^\s*ic\s*\d{1,3}\s*$|^\s*i/?c\b"),
    # taxes (customs duties are procurement-related logistics cost and are NOT excluded)
    ("tax", "text", r"\b(sales|use|input|output|withholding|income|property|payroll|excise|stamp|corporate|corporation|"
             r"local|state|municipal|franchise|value[\s\-]added|gross\s+receipts|business)\s+tax(es)?\b"),
    ("tax", "text", r"\btax(es)?\s+(expense|payment|payable|liabilit|due|assessment|return|penalt|interest|levy)"),
    ("tax", "text", r"^\s*(vat|gst|hst|pst|tax|taxes|iva|tva|mwst|ust)\s*[-:/]?\s*$"),
    ("tax", "text", r"\b(vat|gst|hst)\s+(payment|payable|input|output|return|adjust|refund|control|on\s+import)"),
    ("tax_authority", "text", r"\b(receiver\s+of\s+revenue|revenue\s+service|tax\s+authorit|internal\s+revenue|hmrc|finanzamt)\b"),
    # financing
    ("financing", "text", r"\binterest\s+(expense|charge|paid|payment|on\s+(loan|overdraft|debt))|\bbank\s+(charge|fee|commission)s?\b|"
             r"\bloan\s+(repayment|interest|principal)|\bfinanc(e|ing)\s+(charge|cost|fee)s?\b|\bdividend|"
             r"\b(fx|forex|exchange(\s+rate)?)\s+(loss|gain|difference|revaluation)|\boverdraft\b"),
    # royalties, revenue, partner billings
    ("royalty_revenue", "text", r"\broyalt(y|ies)\b|\bpartner\s+(billing|share|payment|distribution)|\bprofit\s+(share|sharing)\b|"
             r"\brevenue\s+(share|sharing|recogn)|\bcommission\s+income\b|\bsales\s+revenue\b"),
    # people costs paid directly (not via a supplier)
    ("payroll", "text", r"\b(payroll|salar(y|ies)|net\s+pay|pension\s+contribution|social\s+security\s+contribution)\b|"
                        r"\bwages\s*(and|&)\s*salar"),
    # accounting entries, refunds to customers, donations: money moves but nothing is bought
    ("accounting", "text", r"\b(depreciation|amorti[sz]ation|impairment)\b|\b(customer|client)\s+refund|"
                           r"\brefund\s+to\s+(customer|client)|\bdonations?\b|\bcharitable\b"),
    # government levies / fines
    ("government_levy", "text", r"\b(skills\s+development\s+levy|municipal\s+rates|traffic\s+fine|penalty\s+interest)\b"),
]

# A service *about* tax or payroll (advice, software, outsourced payroll) is bought from a supplier:
# these patterns cancel tax / payroll hits.
DEFAULT_ALLOW = [
    ({"tax", "payroll"}, "text",
     r"\b(assist\w*|advi[cs]\w*|consult\w*|services?|prepar\w*|complian\w*|software|filing|audit\w*|provider|outsourc\w*|bureau)\b.{0,40}\b(tax|payroll)|"
     r"\b(tax|payroll)\b.{0,40}\b(assist\w*|advi[cs]\w*|consult\w*|services?|software|complian\w*|filing|provider|outsourc\w*|bureau|system)\b"),
]


def map_status(values: pd.Series, cfg: dict) -> pd.Series:
    sc = cfg["status"]
    exact = {str(k).strip().lower(): v for k, v in (sc.get("map") or {}).items()}
    pats = [(re.compile(p, re.I), r) for p, r in DEFAULT_STATUS_PATTERNS] if sc.get("use_default_patterns", True) else []
    cache = {}
    for v in values.unique():
        s = str(v).strip().lower()
        if s in exact:
            r = exact[s]
        elif not s:
            r = sc.get("blank", INCLUDE)
        else:
            r = next((res for rx, res in pats if rx.search(s)), sc.get("unknown", "NOT_APPROVED"))
        r = INCLUDE if str(r).lower() in ("include", "ok", "1", "true") else str(r).upper()
        if r != INCLUDE and r not in REASONS:
            raise ValueError(f"status map value {r!r} must be include or one of {REASONS}")
        cache[v] = r
    if len(cache) <= 40:
        log.info("status mapping: %s", {k: v for k, v in cache.items()})
    return values.map(cache)


def _contains(s: pd.Series, pattern: str) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # capture groups in user patterns are fine
        return s.str.contains(pattern, flags=re.I, regex=True, na=False)


def _field_text(df: pd.DataFrame, group: str) -> pd.Series:
    if group == "gl":
        return df["gl_description"].fillna("")
    if group == "text":
        return (df["description"].fillna("") + " || " + df["gl_description"].fillna("") + " || " +
                df["supplier_name"].fillna("") + " || " + df["document_type"].fillna(""))
    if group in df.columns:
        return df[group].fillna("").astype(str)
    raise KeyError(f"unknown field {group} in non_procurement rule")


def non_procurement(df: pd.DataFrame, cfg: dict, labels: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """Return (is_non_procurement bool series, rule description series)."""
    npc = cfg["non_procurement"]
    hit = pd.Series(False, index=df.index)
    why = pd.Series("", index=df.index, dtype=object)
    rules = []
    if npc.get("use_default_rules", True):
        rules += [{"field": f, "pattern": p, "name": f"default:{n}"} for n, f, p in DEFAULT_NON_PROCUREMENT]
    for i, r in enumerate(npc.get("rules") or []):
        rules.append({"field": r.get("field", "text"), "pattern": r["pattern"], "name": r.get("name", f"config:{i}"),
                      "exclude": r.get("exclude", True)})
    cache: dict[str, pd.Series] = {}
    for r in rules:
        f = r["field"]
        if f not in cache:
            cache[f] = _field_text(df, f)
        m = _contains(cache[f], r["pattern"])
        if r.get("exclude", True) is False:   # an allow-rule overrides earlier hits
            hit &= ~m
            why[m] = ""
            continue
        new = m & ~hit
        why[new] = r["name"]
        hit |= m
    if npc.get("use_default_rules", True):
        for applies, f, pat in DEFAULT_ALLOW:
            if f not in cache:
                cache[f] = _field_text(df, f)
            m = hit & why.isin([f"default:{a}" for a in applies]) & _contains(cache[f], pat)
            hit &= ~m
            why[m] = ""
    # learned from labelled examples: field values that are (almost) always excluded
    if labels is not None and npc.get("learn_from_labels", True) and "exclusion_reason" in labels.columns:
        lab = df[["record_id"]].join(labels[["exclusion_reason"]], on="record_id")["exclusion_reason"]
        known = lab.notna()
        isnp = lab.fillna("").eq("NON_PROCUREMENT")
        for col in ("gl_account", "gl_description", "supplier_id", "document_type"):
            vals = df[col].where(df[col].ne(""))
            t = pd.DataFrame({"v": vals[known], "np": isnp[known]}).dropna()
            if t.empty:
                continue
            g = t.groupby("v")["np"].agg(["mean", "size"])
            bad = g[(g["size"] >= 10) & (g["mean"] >= 0.98)].index
            if len(bad):
                m = df[col].isin(bad) & ~hit
                why[m] = f"learned:{col}"
                hit |= m
                log.info("non-procurement learned from labels on %s: %d values, %d rows", col, len(bad), int(m.sum()))
    # supplier propagation: a supplier whose lines are nearly all non-procurement is non-procurement
    sp = npc.get("supplier_propagation") or {}
    if sp.get("enabled", True):
        sid = np.where(df["supplier_id"].ne(""), df["supplier_id"], df["supplier_name"].str.upper().str.strip())
        t = pd.DataFrame({"s": sid, "h": hit})
        t = t[t.s != ""]
        g = t.groupby("s")["h"].agg(["mean", "size"])
        bad = g[(g["size"] >= int(sp.get("min_lines", 3))) & (g["mean"] >= float(sp.get("min_share", 0.8)))].index
        m = pd.Series(sid, index=df.index).isin(bad) & ~hit
        if m.any():
            log.info("non-procurement supplier propagation: %d suppliers, %d extra rows", len(bad), int(m.sum()))
        why[m] = "supplier_propagation"
        hit |= m
    return hit, why
