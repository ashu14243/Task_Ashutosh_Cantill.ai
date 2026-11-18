import math
from datetime import date

import pandas as pd
import pytest

from spendtool.normalize import fix_ocr_token, norm_text, parse_amount, parse_date, parse_rate, supplier_core


@pytest.mark.parametrize("raw,expected", [
    ("1,234.56", 1234.56), ("1.234,56", 1234.56), ("(1 234.56)", -1234.56), ("1234.56-", -1234.56),
    ("R 1 234", 1234.0), ("-5,266.63", -5266.63), ("12,5", 12.5), ("1,234", 1234.0), ("1.234.567", 1234567.0),
    ("USD 12.00", 12.0), ("", math.nan), ("abc", math.nan), (17, 17.0), ("1'234.50", 1234.5),
])
def test_parse_amount(raw, expected):
    got = parse_amount(raw)
    assert (math.isnan(got) and math.isnan(expected)) or got == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [("15", 0.15), ("15%", 0.15), ("0.15", 0.15), ("0", 0.0), ("", 0.15), (None, 0.15), ("7.5 %", 0.075)])
def test_parse_rate(raw, expected):
    assert parse_rate(raw, default=0.15) == pytest.approx(expected)


@pytest.mark.parametrize("raw,dayfirst,expected", [
    ("2024-03-29", True, date(2024, 3, 29)), ("27-Mar-24", True, date(2024, 3, 27)),
    ("11/10/2022", True, date(2022, 10, 11)), ("11/10/2022", False, date(2022, 11, 10)),
    ("25/05/2024", False, date(2024, 5, 25)),  # unambiguous: day > 12 wins over the default
    ("20240115", True, date(2024, 1, 15)), ("45292", True, date(2024, 1, 1)), ("15.01.2024", True, date(2024, 1, 15)),
    ("Jan 5, 2024", True, date(2024, 1, 5)), ("31/02/2024", True, None), ("garbage", True, None),
])
def test_parse_date(raw, dayfirst, expected):
    d, flag = parse_date(raw, dayfirst=dayfirst)
    assert d == expected
    assert (flag == "") == (expected is not None)


def test_ocr_and_text():
    assert fix_ocr_token("P0WDER") == "POWDER"
    assert fix_ocr_token("RE5IN") == "RESIN"
    assert fix_ocr_token("20MM") == "20MM"          # units untouched
    assert fix_ocr_token("A2U90") == "A2U90"        # codes with several digits untouched
    assert norm_text("R0D ANCHR;DIA:20MM") == "ROD ANCHR DIA 20MM"


@pytest.mark.parametrize("a,b", [
    ("OAK ENT MINING SA (PTY) LTD", "OAKENT MINING SA (PTY) LTD"), ("KEL MONT  (PTY) LTD", "Kelmont (Pty) Ltd"),
    ("AS HAX (PTY) LTD", "ASHAX (PTY) LTD"), ("PELSTONE PTY LTD", "Pelstone (Pty) Ltd"), ("Siemens AG", "SIEMENS"),
])
def test_supplier_core_equal(a, b):
    assert supplier_core(a) == supplier_core(b)


def test_supplier_core_distinct():
    assert supplier_core("DUNAX (PTY) LTD") != supplier_core("DUNVALE MINING")
