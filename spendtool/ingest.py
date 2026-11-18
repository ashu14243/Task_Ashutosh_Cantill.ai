"""Read every source in a config into one canonical DataFrame.

Field mapping syntax (per source, under `fields:`):
  amount: value_zar                 # a column
  description: [material_short_text, service_short_text]   # several columns, joined with ' | '
  currency: {value: ZAR}            # a constant
  po_number: {column: po_reference, pattern: '(\\d{10})'}  # regex extract (group 1)
  po_line: {column: po_reference, pattern: '/(\\d+)', type: int}
"""
from __future__ import annotations

import csv
import logging
import re

import numpy as np
import pandas as pd

from .config import resolve_path

log = logging.getLogger(__name__)

TEXT_JOIN = " | "


def _read_table(path, read_opts: dict) -> pd.DataFrame:
    p = str(path)
    opts = dict(read_opts or {})
    if p.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p, dtype=str, sheet_name=opts.get("sheet", 0)).fillna("")
    enc = opts.pop("encoding", None)
    sep = opts.pop("delimiter", opts.pop("sep", None))
    if sep is None:
        with open(p, "r", encoding=enc or "utf-8", errors="replace") as fh:
            sample = fh.read(65536)
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
    for e in ([enc] if enc else ["utf-8-sig", "cp1252", "latin-1"]):
        try:
            return pd.read_csv(p, dtype=str, keep_default_na=False, sep=sep, encoding=e,
                               on_bad_lines="warn", **opts)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode {p}")


def _get_field(df: pd.DataFrame, spec, name: str) -> pd.Series | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec not in df.columns:
            raise KeyError(f"column '{spec}' (field {name}) not in file; columns: {list(df.columns)}")
        return df[spec].astype(str).str.strip()
    if isinstance(spec, list):
        parts = [_get_field(df, c, name) for c in spec]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        out = parts[0].copy()
        for p in parts[1:]:
            both = out.ne("") & p.ne("") & (out.str.upper() != p.str.upper())
            out = np.where(both, out + TEXT_JOIN + p, np.where(out.eq(""), p, out))
            out = pd.Series(out, index=df.index)
        return out
    if isinstance(spec, dict):
        if "value" in spec:
            return pd.Series([str(spec["value"])] * len(df), index=df.index)
        col = _get_field(df, spec.get("column") or spec.get("columns"), name)
        if col is None:
            return None
        if spec.get("pattern"):
            rx = re.compile(spec["pattern"], re.I)
            col = col.map(lambda v: (m.group(1) if m.groups() else m.group(0)) if (m := rx.search(v)) else "")
        if spec.get("strip_leading_zeros"):
            col = col.str.lstrip("0")
        if spec.get("type") == "int":
            col = col.map(lambda v: str(int(float(v))) if re.fullmatch(r"-?\d+(\.\d+)?", v or "") else "")
        if spec.get("map"):
            m = {str(k).upper(): str(v) for k, v in spec["map"].items()}
            col = col.map(lambda v: m.get(v.upper(), v))
        return col
    raise ValueError(f"bad field spec for {name}: {spec!r}")


def load_sources(cfg: dict) -> pd.DataFrame:
    frames = []
    for order, src in enumerate(cfg["sources"]):
        path = resolve_path(cfg, src["file"])
        if not path.exists():
            raise FileNotFoundError(f"source {src['name']}: file not found: {path}")
        raw = _read_table(path, src.get("read", {}))
        if src.get("filter"):
            raw = raw.query(src["filter"], engine="python")
        log.info("source %s: %d rows from %s", src["name"], len(raw), path)
        out = pd.DataFrame(index=raw.index)
        fields = src["fields"]
        for f, spec in fields.items():
            out[f] = _get_field(raw, spec, f)
        idfmt = src.get("id_format", "{id}")
        out["record_id"] = [idfmt.format(id=v) for v in out["id"]]
        out["source"] = src["name"]
        out["source_order"] = order
        out["row_in_source"] = np.arange(len(out))
        out["amount_type"] = src.get("amount_type", "net")
        out["default_tax_rate"] = float(src.get("default_tax_rate", 0.0) or 0.0)
        out["status_present"] = "status" in fields
        out["_raw_index"] = raw.index
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    for c in ["amount", "tax_rate", "tax_amount", "currency", "fx_rate", "date", "fx_date", "supplier_name",
              "supplier_id", "description", "item_code", "quantity", "unit_price", "po_number", "po_line",
              "invoice_number", "invoice_line", "status", "gl_account", "gl_description", "cost_centre",
              "document_type", "extra_text"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    dup = df["record_id"].duplicated(keep=False)
    if dup.any():
        # keep the source IDs untouched (the scorer joins on them); every input row still gets
        # exactly one output row, so a repeated ID appears as many times as it does in the input
        log.warning("%d rows share a record_id with another row; IDs kept as in the source", dup.sum())
    df["_uid"] = np.arange(len(df))
    return df
