"""Amounts: parse, remove tax, convert to reporting currency. Dates: parse and flag."""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .config import resolve_path
from .normalize import parse_amount_series, parse_date_series, parse_rate
from .ingest import _read_table

log = logging.getLogger(__name__)


def parse_dates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dc = cfg["dates"]
    for col in ("date", "fx_date"):
        d, f = parse_date_series(df[col], dayfirst=dc.get("dayfirst", True), formats=dc.get("formats") or [],
                                 min_year=dc.get("min_year", 2000), max_year=dc.get("max_year", 2035))
        df[col + "_parsed"] = d
        df[col + "_flag"] = f
    df.loc[df["fx_date"].eq(""), "fx_date_flag"] = "BLANK"
    return df


def compute_net(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dec = cfg["amounts"].get("decimal", "auto")
    df["amount_num"] = parse_amount_series(df["amount"], dec)
    df["amount_flag"] = np.where(df["amount_num"].isna(), "UNPARSEABLE_AMOUNT", "")
    gross = df["amount_type"].eq("gross")
    rate = pd.Series(np.nan, index=df.index)
    if gross.any():
        rates = [parse_rate(v, default=d) for v, d in zip(df.loc[gross, "tax_rate"], df.loc[gross, "default_tax_rate"])]
        rate[gross] = rates
        tax_amt = parse_amount_series(df["tax_amount"], dec)
        use_amt = gross & tax_amt.notna() & df["tax_amount"].ne("")
        net = df["amount_num"].copy()
        net[gross] = df.loc[gross, "amount_num"] / (1.0 + rate[gross].fillna(0.0))
        net[use_amt] = df.loc[use_amt, "amount_num"] - tax_amt[use_amt]
        df["net_doc"] = net
    else:
        df["net_doc"] = df["amount_num"]
    df["tax_rate_used"] = rate
    return df


class FxTable:
    def __init__(self, cfg: dict):
        fxc = cfg.get("fx") or {}
        self.direction = fxc.get("direction", "units_per_reporting")
        self.table: dict[str, dict[str, float]] = {}
        if fxc.get("file"):
            t = _read_table(resolve_path(cfg, fxc["file"]), fxc.get("read", {}))
            cc, mc, rc = fxc.get("currency_column", "currency"), fxc.get("month_column", "month"), fxc.get("rate_column", "rate")
            for c, m, r in zip(t[cc], t[mc], t[rc]):
                try:
                    r = float(str(r).replace(",", ""))
                except ValueError:
                    continue
                if r > 0:
                    self.table.setdefault(str(c).strip().upper(), {})[str(m).strip()[:7]] = r
        self._months = {c: sorted(v) for c, v in self.table.items()}

    def lookup(self, cur: str, month: str | None):
        """Return (rate, how). Exact month, else nearest month for that currency."""
        t = self.table.get(cur)
        if not t:
            return math.nan, "NO_FX_FOR_CURRENCY"
        if month and month in t:
            return t[month], "TABLE_MONTH"
        ms = self._months[cur]
        if not month:
            return t[ms[-1]], "TABLE_LATEST"
        # nearest month by absolute distance
        def key(x):
            y1, m1 = int(x[:4]), int(x[5:7])
            y2, m2 = int(month[:4]), int(month[5:7])
            return abs((y1 * 12 + m1) - (y2 * 12 + m2))
        best = min(ms, key=key)
        return t[best], "TABLE_NEAREST_MONTH"


def convert_currency(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rc = cfg["reporting_currency"].upper()
    fxc = cfg.get("fx") or {}
    inline_dir = fxc.get("inline_rate_direction", "units_per_reporting")
    table = FxTable(cfg)
    cur = df["currency"].str.strip().str.upper().replace("", rc)
    df["currency_norm"] = cur
    inline = parse_amount_series(df["fx_rate"])
    rates = np.full(len(df), np.nan)
    how = np.array([""] * len(df), dtype=object)
    same = cur.eq(rc).to_numpy()
    rates[same] = 1.0
    how[same] = "REPORTING_CCY"
    ok_inline = (~same) & inline.notna().to_numpy() & (inline.to_numpy() > 0)
    rates[ok_inline] = inline.to_numpy()[ok_inline]
    how[ok_inline] = "INLINE"
    need = (~same) & (~ok_inline)
    if need.any():
        # month: fx_date if plausible, else date, else none
        fxm = df["fx_date_parsed"].dt.strftime("%Y-%m")
        fxm = fxm.where(df["fx_date_flag"].eq(""), None)
        dm = df["date_parsed"].dt.strftime("%Y-%m").where(df["date_flag"].eq(""), None)
        month = fxm.fillna(dm)
        cache = {}
        for i in np.where(need)[0]:
            k = (cur.iat[i], month.iat[i])
            if k not in cache:
                cache[k] = table.lookup(k[0], k[1] if isinstance(k[1], str) else None)
            rates[i], how[i] = cache[k]
    df["fx_rate_used"] = rates
    df["fx_how"] = how
    # convert
    spend = df["net_doc"].to_numpy(dtype=float).copy()
    inl = how == "INLINE"
    tab = np.isin(how, ["TABLE_MONTH", "TABLE_NEAREST_MONTH", "TABLE_LATEST"])
    def conv(amount, rate, direction):
        return amount / rate if direction == "units_per_reporting" else amount * rate
    spend[inl] = conv(spend[inl], rates[inl], inline_dir)
    spend[tab] = conv(spend[tab], rates[tab], table.direction)
    bad = ~(same | inl | tab) | np.isnan(rates)
    spend[bad & ~same] = np.nan
    df["spend_rc"] = spend
    n_bad = int(np.sum(np.isnan(spend) & ~np.isnan(df["net_doc"].to_numpy(dtype=float))))
    if n_bad:
        log.warning("%d rows have no usable FX rate; their spend is left at 0 and flagged NO_FX", n_bad)
    df["fx_flag"] = np.where(np.isnan(spend) & ~np.isnan(df["net_doc"].to_numpy(dtype=float)), "NO_FX", "")
    return df
