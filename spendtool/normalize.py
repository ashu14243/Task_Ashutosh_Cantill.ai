"""Parsing and normalisation helpers: amounts, rates, dates, text, names.

All functions are defensive: bad input returns NaN / None plus a flag, never raises.
"""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- amounts
_CURRENCY_JUNK = re.compile(r"[^\d,.\-()+eE]")


def parse_amount(value, decimal: str = "auto") -> float:
    """Parse '1,234.56', '1.234,56', '(1 234.56)', '1234.56-', 'R 1 234', 'USD 12' ...

    decimal: 'auto' | '.' | ','
    """
    if value is None:
        return math.nan
    if isinstance(value, (int, float, np.number)):
        return float(value) if not (isinstance(value, float) and math.isnan(value)) else math.nan
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "-", "n/a", "na"}:
        return math.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    s = s.replace("−", "-").replace(" ", "").replace(" ", "").replace("'", "")
    if s.endswith("-") and not s.startswith("-"):
        neg, s = True, s[:-1]
    s = _CURRENCY_JUNK.sub("", s)
    if s.startswith("-"):
        neg, s = (not neg), s[1:]
    s = s.lstrip("+")
    if not s:
        return math.nan
    has_c, has_d = "," in s, "." in s
    if decimal == ",":
        s = s.replace(".", "").replace(",", ".")
    elif decimal == ".":
        s = s.replace(",", "")
    elif has_c and has_d:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_c:
        # thousands pattern 1,234 or 1,234,567 -> thousands; else decimal comma
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".") if s.count(",") == 1 else s.replace(",", "")
    elif has_d and s.count(".") > 1:
        s = s.replace(".", "")  # 1.234.567 -> thousands dots
    try:
        v = float(s)
    except ValueError:
        return math.nan
    return -v if neg else v


def parse_amount_series(s: pd.Series, decimal: str = "auto") -> pd.Series:
    # fast path for clean numeric columns
    num = pd.to_numeric(s, errors="coerce")
    bad = num.isna() & s.astype(str).str.strip().ne("") & s.notna()
    if bad.any():
        num = num.astype(float)
        num[bad] = [parse_amount(v, decimal) for v in s[bad]]
    return num.astype(float)


def parse_rate(value, default: float | None = None) -> float:
    """VAT/tax rate: '15', '15%', '0.15', '15.0 %' -> 0.15. Blank -> default."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default if default is not None else math.nan
    s = str(value).strip()
    if not s:
        return default if default is not None else math.nan
    pct = "%" in s
    v = parse_amount(s.replace("%", ""))
    if math.isnan(v):
        return default if default is not None else math.nan
    if pct or v > 1.0:
        v = v / 100.0
    return v


# --------------------------------------------------------------------------- dates
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTHS.update({"sept": 9, "janv": 1, "fev": 2, "fevr": 2, "mars": 3, "avr": 4, "mai": 5, "juin": 6,
                "juil": 7, "aout": 8, "dez": 12, "okt": 10, "mrz": 3, "maerz": 3, "ene": 1, "abr": 4,
                "ago": 8, "set": 9, "dic": 12, "gen": 1, "mag": 5, "giu": 6, "lug": 7, "ott": 10})


def _mk(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def _year(y: str) -> int:
    y = int(y)
    if y < 100:
        y += 2000 if y < 70 else 1900
    return y


def parse_date(value, dayfirst: bool = True, formats: list[str] | None = None):
    """Return (date | None, flag). flag in {'', 'BLANK', 'UNPARSEABLE'}."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, "BLANK"
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date(), ""
    if isinstance(value, date):
        return value, ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none", "null", "0", "00000000"}:
        return None, "BLANK"
    for f in formats or []:
        try:
            return datetime.strptime(s, f).date(), ""
        except ValueError:
            pass
    s2 = s.split("T")[0].split(" ")[0] if re.match(r"^\d{4}-\d\d-\d\d[T ]", s) else s
    # ISO yyyy-mm-dd / yyyy/mm/dd / yyyy.mm.dd
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s2)
    if m:
        d = _mk(m[1], m[2], m[3])
        return (d, "") if d else (None, "UNPARSEABLE")
    # yyyymmdd
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s2)
    if m and 1900 < int(m[1]) < 2100:
        d = _mk(m[1], m[2], m[3])
        return (d, "") if d else (None, "UNPARSEABLE")
    # dd/mm/yyyy or mm/dd/yyyy (also - and .)
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", s2)
    if m:
        a, b, y = int(m[1]), int(m[2]), _year(m[3])
        if a > 12 >= b:
            d = _mk(y, b, a)
        elif b > 12 >= a:
            d = _mk(y, a, b)
        else:
            d = _mk(y, b, a) if dayfirst else _mk(y, a, b)
        return (d, "") if d else (None, "UNPARSEABLE")
    # dd-Mon-yy / dd Mon yyyy / Mon dd, yyyy
    m = re.fullmatch(r"(\d{1,2})[-/ .]?([A-Za-zÀ-ſ]{3,9})\.?[-/ .,]?\s*(\d{2,4})", s2)
    if m:
        mon = _MONTHS.get(_strip_accents(m[2]).lower()[:4]) or _MONTHS.get(_strip_accents(m[2]).lower()[:3])
        if mon:
            d = _mk(_year(m[3]), mon, m[1])
            return (d, "") if d else (None, "UNPARSEABLE")
    m = re.fullmatch(r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{2,4})", s)
    if m:
        mon = _MONTHS.get(m[1].lower()[:4]) or _MONTHS.get(m[1].lower()[:3])
        if mon:
            d = _mk(_year(m[3]), mon, m[2])
            return (d, "") if d else (None, "UNPARSEABLE")
    # Excel serial
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        v = float(s)
        if 20000 < v < 80000:
            return (date(1899, 12, 30) + timedelta(days=int(v))), ""
    return None, "UNPARSEABLE"


def parse_date_series(s: pd.Series, dayfirst=True, formats=None, min_year=2000, max_year=2035):
    """Vectorised over unique values. Returns (datetime64 series, flag series)."""
    uniq = pd.unique(s.astype(str))
    cache = {}
    for u in uniq:
        d, f = parse_date(u, dayfirst, formats)
        if d is not None and not (min_year <= d.year <= max_year):
            f = "IMPLAUSIBLE"
        cache[u] = (d, f)
    keys = s.astype(str)
    dates = pd.to_datetime(keys.map(lambda k: cache[k][0]), errors="coerce")
    flags = keys.map(lambda k: cache[k][1])
    return dates, flags


# --------------------------------------------------------------------------- text
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


_OCR_DIGIT_TO_ALPHA = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "3": "E", "4": "A", "6": "G", "7": "T"})
_ALPHA = re.compile(r"[A-Z]")


def fix_ocr_token(tok: str) -> str:
    """Fix digit-for-letter OCR errors inside mostly-alphabetic tokens: 'P0WDER' -> 'POWDER',
    'RE5IN' -> 'RESIN', '8OLT' -> 'BOLT'. Pure numbers and codes with >=2 digits untouched."""
    if not tok or not _ALPHA.search(tok):
        return tok
    n_digit = sum(c.isdigit() for c in tok)
    n_alpha = sum(c.isalpha() for c in tok)
    if n_digit == 0 or n_digit > 1 or n_alpha < 2 or len(tok) < 3:
        return tok
    # common unit forms like 20MM, 4X4, M12 are kept
    if re.fullmatch(r"\d+[A-Z]{1,3}|[A-Z]{1,2}\d+", tok):
        return tok
    return tok.translate(_OCR_DIGIT_TO_ALPHA)


_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)


def norm_text(s, ocr: bool = True) -> str:
    """Uppercase, strip accents, punctuation -> space, OCR-fix tokens, collapse spaces."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = _strip_accents(str(s)).upper()
    toks = [t for t in _TOKEN_SPLIT.split(s.replace("_", " ")) if t]
    if ocr:
        toks = [fix_ocr_token(t) for t in toks]
    return " ".join(toks)


# --------------------------------------------------------------------------- supplier names
LEGAL_SUFFIXES = {
    "PTY", "PTYLTD", "LTD", "LIMITED", "CC", "INC", "INCORPORATED", "LLC", "LLP", "LP", "GMBH", "AG", "KG",
    "SA", "SAS", "SARL", "SRL", "SPA", "BV", "NV", "PLC", "CO", "CORP", "CORPORATION", "COMPANY", "THE",
    "AB", "AS", "OY", "SL", "SDN", "BHD", "PVT", "PRIVATE", "KFT", "SRO", "SP", "ZOO", "DE", "CV", "LTDA",
    "EIRELI", "ME", "EPP", "OOO", "ZAO", "PJSC", "JSC", "TBK", "PT", "KK", "GROUP", "HOLDINGS", "AND", "&",
    "PROPRIETARY", "EN", "Y", "CIA", "S", "A", "L", "C", "B", "V", "N", "U",
}


def supplier_core(name) -> str:
    """Canonical compact form for supplier dedup: 'Oak Ent Mining SA (Pty) Ltd' -> 'OAKENTMINING'."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    raw = str(name)
    # parenthesised legal forms: '(Pty)', '(Pvt.)'
    raw = re.sub(r"\((\s*[A-Za-z.]{1,12}\s*)\)", lambda m: " " if norm_text(m.group(1)).replace(" ", "") in LEGAL_SUFFIXES else m.group(0), raw)
    s = norm_text(raw)
    if not s:
        return ""
    toks = s.split()
    # strip legal forms only at the end (and a leading 'THE'), so 'AS HAX' keeps its 'AS'
    while len(toks) > 1 and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    if len(toks) > 1 and toks[0] == "THE":
        toks.pop(0)
    return "".join(toks)
