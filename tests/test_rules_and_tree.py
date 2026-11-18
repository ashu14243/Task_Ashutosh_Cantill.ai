import pandas as pd

from spendtool.classify.lexicon import stem
from spendtool.classify.text import tokens
from spendtool.config import DEFAULTS, _deep_merge
from spendtool.spend_rules import map_status, non_procurement


def _cfg(**kw):
    return _deep_merge(DEFAULTS, kw)


def test_status_defaults():
    vals = pd.Series(["Paid", "Approved", "Cancelled", "On Hold", "Rejected", "Submitted", "Disputed", "Void",
                      "Approved - awaiting payment", "Unpaid", "Not approved", "Posted", ""])
    got = map_status(vals, _cfg()).tolist()
    assert got == ["include", "include", "CANCELLED", "NOT_APPROVED", "NOT_APPROVED", "NOT_APPROVED", "NOT_APPROVED",
                   "CANCELLED", "include", "NOT_APPROVED", "NOT_APPROVED", "include", "include"]


def test_status_config_map_wins():
    got = map_status(pd.Series(["Open"]), _cfg(status={"map": {"open": "include"}})).tolist()
    assert got == ["include"]


def _frame(desc, gl=""):
    n = len(desc)
    return pd.DataFrame({"description": desc, "gl_description": gl if isinstance(gl, list) else [gl] * n,
                         "supplier_name": [""] * n, "supplier_id": [f"V{i}" for i in range(n)],
                         "document_type": [""] * n, "gl_account": [""] * n, "record_id": [str(i) for i in range(n)]})


def test_non_procurement_defaults():
    df = _frame(["State & Local Taxes - ", "Sales Tax Expense - ", "GST Input Tax -", "Freight - ", "Import duty",
                 "Assistance with Property Tax v -", "GlobalView Payroll Services -", "Interest on loan", "Royalties Q3",
                 "Bearing 6205", ""],
                ["PEUS"] * 10 + ["IC Payables"])
    hit, why = non_procurement(df, _cfg())
    assert hit.tolist() == [True, True, True, False, False, False, False, True, True, False, True]


def test_ic_gl_codes():
    df = _frame(["", "", "", ""], ["IC06", "I/C Payable", "Intercompany - CZ", "AP - Other"])
    hit, _ = non_procurement(df, _cfg())
    assert hit.tolist() == [True, True, True, False]


def test_tokens_and_stem():
    assert stem("bearings") == "bearing" and stem("boxes") == "box" and stem("batteries") == "battery"
    t = tokens("VehicleMaintRepairSV")
    assert t[:3] == ("vehicle", "maintenance", "repair")
    assert "powder" in tokens("P0WDER;STNEDST")
