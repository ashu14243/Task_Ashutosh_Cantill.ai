"""A tiny unseen 'company C' built in a temp folder: one invoice file, a 4-level tree, no labels."""
import os

import pandas as pd
import yaml

from spendtool.config import load_config
from spendtool.pipeline import OUTPUT_COLUMNS, run

TREE = [
    ("Indirect", "IT", "Hardware", "Laptops & Desktops"),
    ("Indirect", "IT", "Software", "Licences & Subscriptions"),
    ("Indirect", "Facilities", "Services", "Cleaning"),
    ("Indirect", "Facilities", "Services", "Security Guarding"),
    ("Indirect", "Logistics", "Transport", "Courier"),
    ("Direct", "Raw Materials", "Metals", "Steel"),
    ("Direct", "Packaging", "Secondary", "Pallets"),
]

ROWS = [
    # id, supplier, desc, gross, vat, ccy, status, date, inv
    ("C1", "Dell Computers Ltd", "Latitude 5440 laptop", "1,150.00", "15%", "EUR", "Paid", "05/03/2024", "INV-1"),
    ("C2", "DELL COMPUTERS LIMITED", "Latitude 5440 laptop", "1150.00", "0.15", "EUR", "Paid", "05/03/2024", "INV-1"),  # exported twice
    ("C3", "SparkleClean Services", "Monthly office cleaning contract", "2300", "", "USD", "Approved", "2024-04-01", "77"),
    ("C4", "Guardian Security", "Security guards night shift", "4600", "15", "USD", "Rejected", "2024-04-02", "78"),
    ("C5", "Microsoft Ireland", "Office 365 E3 subscription renewal", "11500", "15%", "USD", "Paid", "2024-04-03", "79"),
    ("C6", "DHL Express", "Courier charges March", "115", "15%", "GBP", "Cancelled", "2024-03-31", "80"),
    ("C7", "ACME Steel Works", "Hot rolled steel coil 2mm", "23000", "15%", "USD", "Paid", "2024-02-10", "81"),
    ("C8", "Pallet Masters", "Wooden pallets 1200x1000", "1150", "15%", "USD", "Paid", "2024-02-11", "82"),
    ("C9", "Acme Group Holdings", "Intercompany recharge Q1", "5000", "0", "USD", "Paid", "2024-02-12", "83"),
    ("C10", "Revenue Service", "VAT payment March", "999", "0", "USD", "Paid", "2024-02-12", "84"),
]


def _make(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    pd.DataFrame(ROWS, columns=["line_id", "vendor_name", "item_description", "gross_amount", "vat_rate", "currency",
                                "invoice_status", "invoice_date", "invoice_no"]).to_csv(d / "inv.csv", index=False)
    pd.DataFrame(TREE, columns=["level_1", "level_2", "level_3", "level_4"]).to_csv(d / "tree.csv", index=False)
    pd.DataFrame([("EUR", "2024-03", 0.9), ("GBP", "2024-03", 0.8)], columns=["ccy", "month", "rate"]).to_csv(d / "fx.csv", index=False)
    cfg = {
        "company": "TESTC", "reporting_currency": "USD", "data_dir": str(d),
        "taxonomy": {"file": "tree.csv"},
        "fx": {"file": "fx.csv", "currency_column": "ccy", "month_column": "month", "rate_column": "rate"},
        "sources": [{"name": "inv", "file": "inv.csv", "amount_type": "gross", "default_tax_rate": 0.15,
                     "fields": {"id": "line_id", "amount": "gross_amount", "tax_rate": "vat_rate", "currency": "currency",
                                "status": "invoice_status", "date": "invoice_date", "supplier_name": "vendor_name",
                                "description": "item_description", "invoice_number": "invoice_no"}}],
        "duplicates": {"within_source": [{"source": "inv", "keys": ["supplier_key", "_inv", "_amt2"]}]},
        "output": {"dir": str(tmp_path / "out")},
        "classification": {"llm": {"enabled": False}},
    }
    (tmp_path / "configs").mkdir()
    p = tmp_path / "configs" / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_unseen_company(tmp_path):
    os.environ["SPENDTOOL_DISABLE_LLM"] = "1"
    cfg = load_config(_make(tmp_path))
    res = run(cfg)
    out = res["frame"].set_index("record_id")
    assert list(res["frame"].columns) == OUTPUT_COLUMNS
    assert len(out) == len(ROWS)
    r = out["exclusion_reason"].to_dict()
    assert r["C2"] == "DUPLICATE" and r["C4"] == "NOT_APPROVED" and r["C6"] == "CANCELLED"
    assert r["C9"] == "NON_PROCUREMENT" and r["C10"] == "NON_PROCUREMENT"
    assert out.loc["C1", "supplier_key"] == out.loc["C2", "supplier_key"]
    # 1150 gross incl 15% VAT = 1000 EUR net; 0.9 EUR per USD -> 1111.11 USD
    assert abs(out.loc["C1", "spend"] - 1111.11) < 0.01
    assert abs(out.loc["C3", "spend"] - 2000.00) < 0.01   # blank VAT -> default 15%
    assert (out.loc[out["include"] == 0, "spend"] == 0).all()
    assert out.loc["C1", "l4"] == "Laptops & Desktops"
    assert out.loc["C5", "l4"] == "Licences & Subscriptions"
    assert out.loc["C3", "l4"] == "Cleaning"
    assert out.loc["C7", "l4"] == "Steel"
    assert out.loc["C8", "l4"] == "Pallets"
    inc = out[out["include"] == 1]
    assert inc[["l1", "l2", "l3", "l4"]].ne("").all().all()
    assert ((inc["confidence"] >= 0) & (inc["confidence"] <= 1)).all()


def test_parent_child_tree(tmp_path):
    from spendtool.classify.taxonomy import load_taxonomy
    t = tmp_path / "pc.csv"
    pd.DataFrame([("1", "", "Indirect"), ("2", "1", "IT"), ("3", "2", "Software"), ("4", "1", "Travel")],
                 columns=["id", "parent_id", "name"]).to_csv(t, index=False)
    tax = load_taxonomy({"taxonomy": {"file": str(t), "max_levels": 4}, "_base_dir": str(tmp_path)})
    assert set(tax.leaves) == {("Indirect", "IT", "Software"), ("Indirect", "Travel")}
