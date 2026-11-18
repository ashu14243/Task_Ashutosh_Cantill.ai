"""Help write a config for an unseen company quickly.

profile_file : prints every column with fill rate, distinct count, value patterns and samples,
               plus what each column looks like it is (amount, date, status ...).
draft_config : writes a commented YAML config from those guesses. Always review it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .ingest import _read_table
from .normalize import parse_amount, parse_date

ROLE_PATTERNS = [
    ("id", r"^(line_?id|record_?id|row_?id|id|line_?no|invoice_?line_?id|transaction_?id|uid)$|record_?id$|(^|_)line_?id$"),
    ("invoice_line", r"^(invoice_?line|line|line_?number|line_?item|item_?no)$"),
    ("invoice_number", r"inv(oice)?_?(number|no|num|nr|ref|id)|^invoice$|document_?(number|no)|^doc_?no"),
    ("po_number", r"^po$|po_?(number|no|num|ref|reference)|purchase_?order"),
    ("po_line", r"po_?(line|item)"),
    ("tax_rate", r"(vat|tax|gst)_?(rate|pct|percent|%)"),
    ("tax_amount", r"(vat|tax|gst)_?(amount|amt|value)$|^(vat|tax|gst)$"),
    ("fx_rate", r"(fx|exchange|currency|conversion)_?rate|^rate$"),
    ("currency", r"^(curr|currency|ccy|cur|currency_?code|doc_?currency)$"),
    ("amount", r"amount|amt|value|net|gross|total|spend|extended|price_?total|line_?total"),
    ("quantity", r"^(qty|quantity|qnty|units)$|quantity"),
    ("unit_price", r"unit_?(price|cost)|price$"),
    ("status", r"status|state|approval|workflow"),
    ("supplier_id", r"(supplier|vendor|creditor|payee)_?(id|no|num|number|code|key)"),
    ("supplier_name", r"supplier|vendor|creditor|payee|merchant"),
    ("gl_description", r"(gl|account|ledger|expense|nominal)_?(desc|description|name|text)|account_?type|expense_?type"),
    ("gl_account", r"^(gl|gl_?account|gl_?code|account|account_?code|account_?no|nominal_?code|ledger_?account)$"),
    ("cost_centre", r"cost_?cent(er|re)|^cc$|department|dept"),
    ("item_code", r"material|item_?(code|no|number|id)|sku|part_?(no|number)|product_?(code|id)|article"),
    ("description", r"desc|text|narrative|memo|item_?name|product_?name|details|line_?item_?desc|service"),
    ("document_type", r"doc(ument)?_?type|transaction_?type|invoice_?type"),
    ("date", r"date|posted|period"),
    ("extra_text", r"commodity|category|class|segment|family|group|uom|unit$|type"),
]


def guess_roles(df: pd.DataFrame) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    taken = set()
    for role, pat in ROLE_PATTERNS:
        for c in df.columns:
            if c in taken:
                continue
            name = c.strip().lower().replace(" ", "_")
            if role in ("supplier_name", "supplier_id", "description") and re.search(r"type|class|categ|group|flag", name):
                continue
            if re.search(pat, name):
                roles.setdefault(role, []).append(c)
                if role not in ("description", "extra_text", "date", "amount"):
                    taken.add(c)
                    break
                taken.add(c)
    return roles


def _pattern(v: str) -> str:
    return re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "A", v))[:25]


def profile_file(path: str, max_values: int = 12) -> None:
    df = _read_table(Path(path), {})
    print(f"\n{path}: {len(df):,} rows x {len(df.columns)} columns\n")
    roles = guess_roles(df)
    rev = {c: r for r, cs in roles.items() for c in cs}
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        nonblank = s[s.ne("")]
        print(f"== {c}   [looks like: {rev.get(c, '?')}]  filled {len(nonblank) / max(len(s), 1):.0%}, distinct {nonblank.nunique():,}")
        if nonblank.empty:
            continue
        pats = nonblank.map(_pattern).value_counts().head(5)
        print("   patterns:", dict(pats))
        vc = nonblank.value_counts()
        if len(vc) <= max_values * 2:
            print("   values:", dict(vc.head(max_values * 2)))
        else:
            print("   top:", dict(vc.head(max_values)))
        num = nonblank.head(2000).map(parse_amount)
        if num.notna().mean() > 0.95:
            print(f"   numeric: min {num.min():,.2f} max {num.max():,.2f} negatives {int((num < 0).sum())}")
        dt = nonblank.head(2000).map(lambda v: parse_date(v)[0])
        if dt.notna().mean() > 0.8:
            first = nonblank.str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}$").dropna().astype(int)
            hint = ""
            if len(first):
                hint = (" -> day first" if (first[0] > 12).any() else "") + (" -> month first" if (first[1] > 12).any() else "")
            print(f"   dates: {dt.min()} .. {dt.max()}{hint}")
    print("\nGuessed roles:")
    for r, cs in roles.items():
        print(f"   {r:15s} <- {cs}")


def draft_config(path: str, taxonomy: str, company: str, currency: str, out: str | None) -> None:
    df = _read_table(Path(path), {})
    roles = guess_roles(df)
    pick = lambda r: (roles.get(r) or [None])[0]  # noqa: E731
    amount_cols = roles.get("amount", [])
    amount = next((c for c in amount_cols if re.search(r"net", c, re.I)), None) or \
        next((c for c in amount_cols if not re.search(r"unit|price|tax|vat", c, re.I)), amount_cols[0] if amount_cols else None)
    gross = bool(amount and re.search(r"gross|incl", amount, re.I))
    desc = roles.get("description", [])
    fields = {
        "id": pick("id") or df.columns[0], "amount": amount, "currency": pick("currency") or {"value": currency},
        "fx_rate": pick("fx_rate"), "tax_rate": pick("tax_rate") if gross else None,
        "tax_amount": pick("tax_amount") if gross else None, "date": (roles.get("date") or [None])[0],
        "supplier_name": pick("supplier_name"), "supplier_id": pick("supplier_id"),
        "description": desc if len(desc) > 1 else (desc[0] if desc else None), "item_code": pick("item_code"),
        "quantity": pick("quantity"), "po_number": pick("po_number"), "po_line": pick("po_line"),
        "invoice_number": pick("invoice_number"), "invoice_line": pick("invoice_line"), "status": pick("status"),
        "gl_account": pick("gl_account"), "gl_description": pick("gl_description"),
        "cost_centre": pick("cost_centre"), "document_type": pick("document_type"),
        "extra_text": roles.get("extra_text") or None,
    }
    fields = {k: v for k, v in fields.items() if v}
    lines = [f"# Draft config for company {company} - generated from {Path(path).name}. REVIEW EVERY LINE.",
             f"company: {company}", "candidate_name: ashutosh", f"reporting_currency: {currency}", "",
             "taxonomy:", f"  file: {taxonomy}", "  max_levels: 4", "",
             "sources:", "  - name: invoices", f"    file: {path}", f"    amount_type: {'gross' if gross else 'net'}"]
    if gross:
        lines.append("    default_tax_rate: 0.0       # set the VAT rate to assume when the rate is blank")
    lines.append("    fields:")
    for k, v in fields.items():
        lines.append(f"      {k}: {v if not isinstance(v, (list, dict)) else repr(v).replace(chr(39), '')}")
    if fields.get("status"):
        vals = df[fields["status"]].astype(str).str.strip().value_counts().index[:30].tolist()
        lines += ["", "status:", "  map:            # every status value seen; check each", ]
        from .spend_rules import map_status
        cfg_stub = {"status": {"map": {}, "use_default_patterns": True, "blank": "include", "unknown": "NOT_APPROVED"}}
        mapped = map_status(pd.Series(vals), cfg_stub)
        for v, m in zip(vals, mapped):
            lines.append(f"    \"{v}\": {m}")
    lines += ["", "non_procurement:", "  use_default_rules: true", "  rules: []      # e.g. - {field: gl_description, pattern: '(?i)intercompany'}",
              "", "duplicates:", "  within_source: []   # e.g. - {source: invoices, keys: [supplier_key, _inv, _amt2]}",
              "", "dates: {dayfirst: true, min_year: 2015, max_year: 2030}", "",
              "classification:", "  text_fields: {description: 3.0, gl_description: 1.0, supplier_name: 1.0, extra_text: 1.0}"]
    target = Path(out or f"configs/company_{company}.yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {target}. Check: amount column & gross/net, currency, status map, id column, date format.")
