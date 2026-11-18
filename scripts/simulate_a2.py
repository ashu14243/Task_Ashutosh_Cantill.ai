"""Simulate the A2 blind test: hold out 20% of A's POs (ERP + tool lines together) as 'new' files,
keep labels only for the rest, run the frozen pipeline the way A2 will be run, score against the
hidden labels.

    python scripts/simulate_a2.py [workdir]      (default /tmp/a2sim)
"""
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendtool.evaluate import bcubed  # noqa: E402

root = Path(__file__).resolve().parents[1]
work = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/a2sim")
work.mkdir(parents=True, exist_ok=True)
A = root / "data" / "company_A"
e = pd.read_csv(A / "erp_goods_receipts.csv", dtype=str, keep_default_na=False)
t = pd.read_csv(A / "invoice_tool_export.csv", dtype=str, keep_default_na=False)
lab = pd.read_csv(A / "labels_A.csv", dtype=str, keep_default_na=False)
held = lambda s: zlib.crc32(s.encode()) % 5 == 0  # noqa: E731
e2 = e.po_number.map(held)
tpo = t.po_reference.str.extract(r"(\d{10})")[0].fillna("")
t2 = np.where(tpo != "", tpo.map(held), t.tool_record_id.map(held))
e[~e2].to_csv(work / "erp_train.csv", index=False)
t[~t2].to_csv(work / "tool_train.csv", index=False)
e[e2].to_csv(work / "erp_a2.csv", index=False)
t[t2].to_csv(work / "tool_a2.csv", index=False)
lab[lab.record_id.isin(set("ERP:" + e[~e2].line_id))].to_csv(work / "labels_train.csv", index=False)
(work / "taxonomy_A.csv").write_bytes((A / "taxonomy_A.csv").read_bytes())

cfg = yaml.safe_load((root / "configs" / "company_A.yaml").read_text())
cfg.update(company="A2SIM", data_dir=str(work))
cfg["taxonomy"]["file"] = "taxonomy_A.csv"
cfg["labels"].update(file="labels_train.csv", training_files={"erp": "erp_train.csv", "tool": "tool_train.csv"})
cfg["sources"][0]["file"], cfg["sources"][1]["file"] = "erp_train.csv", "tool_train.csv"
(work / "configs").mkdir(exist_ok=True)
cp = work / "configs" / "company_A2SIM.yaml"
cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
out = work / "out_a2.csv"
subprocess.check_call([sys.executable, "-m", "spendtool", "run", "--config", str(cp), "--no-llm",
                       "--input", f"erp={work / 'erp_a2.csv'}", "--input", f"tool={work / 'tool_a2.csv'}",
                       "--output", str(out)], cwd=root, env=dict(os.environ))
o = pd.read_csv(out, dtype=str, keep_default_na=False)
m = o.merge(lab, on="record_id", suffixes=("", "_gold"))
w = m.spend.astype(float).abs()
print(f"\nA2 simulation: {int(e2.sum())} ERP + {int(t2.sum())} tool lines held out")
for k in "123":
    ok = m[f"l{k}"] == m[f"category_l{k}"]
    print(f"L{k}: by count {ok.mean():.4f}  by spend {(ok * w).sum() / w.sum():.4f}")
print("supplier", bcubed(m.supplier_key, m.supplier_key_gold))
mi = m[m.item_key_gold != ""]
print("item", bcubed(mi.item_key, mi.item_key_gold))
