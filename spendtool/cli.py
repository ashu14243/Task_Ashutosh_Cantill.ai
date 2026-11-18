"""Command line.

  python -m spendtool run      --config configs/company_A.yaml
  python -m spendtool run      --config configs/company_A.yaml --input erp=new/erp.csv --input tool=new/tool.csv --output submission/x.csv
  python -m spendtool evaluate --config configs/company_A.yaml          # zero-label accuracy vs labels
  python -m spendtool profile  --file some_invoices.csv                   # inspect an unseen file
  python -m spendtool init-config --file invoices.csv --taxonomy tree.csv --company C  # draft a config
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import load_config


def _overrides(args) -> dict:
    ov: dict = {}
    if getattr(args, "data_dir", None):
        ov["data_dir"] = args.data_dir
    return ov


def _apply_inputs(cfg: dict, args) -> None:
    for spec in getattr(args, "input", None) or []:
        if "=" in spec:
            name, path = spec.split("=", 1)
        elif len(cfg["sources"]) == 1:
            name, path = cfg["sources"][0]["name"], spec
        else:
            raise SystemExit(f"--input {spec}: use name=path; sources are {[s['name'] for s in cfg['sources']]}")
        hit = [s for s in cfg["sources"] if s["name"] == name]
        if not hit:
            raise SystemExit(f"--input: no source named {name}")
        hit[0]["file"] = path
    if getattr(args, "taxonomy", None):
        cfg["taxonomy"]["file"] = args.taxonomy


def main(argv=None):
    p = argparse.ArgumentParser(prog="spendtool", description="Config-driven spend cube pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline for one company config")
    r.add_argument("--config", required=True)
    r.add_argument("--input", action="append", help="override a source file: name=path (repeatable)")
    r.add_argument("--taxonomy", help="override the taxonomy file")
    r.add_argument("--output", help="output CSV path (default outputs/<candidate>__<company>.csv)")
    r.add_argument("--data-dir", help="folder that relative data paths in the config resolve against")
    r.add_argument("--no-labels", action="store_true", help="ignore labels (zero-label mode)")
    r.add_argument("--no-llm", action="store_true", help="never call an LLM API")
    r.add_argument("-v", "--verbose", action="store_true")

    e = sub.add_parser("evaluate", help="measure accuracy against the config's labels (labels hidden from the run)")
    e.add_argument("--config", required=True)
    e.add_argument("--data-dir")
    e.add_argument("--with-labels", action="store_true", help="also report a run that uses labels (cross-validated)")
    e.add_argument("--no-llm", action="store_true")

    pr = sub.add_parser("profile", help="summarise an unseen input file to help write a config")
    pr.add_argument("--file", required=True)
    pr.add_argument("--max-values", type=int, default=12)

    ic = sub.add_parser("init-config", help="draft a config for a new single-file company")
    ic.add_argument("--file", required=True)
    ic.add_argument("--taxonomy", required=True)
    ic.add_argument("--company", required=True)
    ic.add_argument("--currency", default="USD")
    ic.add_argument("--out", help="where to write the YAML (default configs/company_<X>.yaml)")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if getattr(args, "no_llm", False):
        import os
        os.environ["SPENDTOOL_DISABLE_LLM"] = "1"

    if args.cmd == "run":
        from .pipeline import run
        cfg = load_config(args.config, _overrides(args))
        _apply_inputs(cfg, args)
        t = time.time()
        res = run(cfg, use_labels=not args.no_labels, output=args.output)
        s = res["summary"]
        print(f"\nwrote {res['output']}")
        print(f"rows={s['rows']} included={s['included']} excluded={s['excluded_by_reason']} "
              f"total_spend={s['total_spend']:,.2f} {s['reporting_currency']} suppliers={s['suppliers']} "
              f"items={s['items']} runtime={time.time() - t:.0f}s")
        if s.get("validation_problems"):
            print("VALIDATION PROBLEMS:", s["validation_problems"])
            return 2
        return 0
    if args.cmd == "evaluate":
        from .evaluate import evaluate
        cfg = load_config(args.config, _overrides(args))
        evaluate(cfg, with_labels=args.with_labels)
        return 0
    if args.cmd == "profile":
        from .profiling import profile_file
        profile_file(args.file, args.max_values)
        return 0
    if args.cmd == "init-config":
        from .profiling import draft_config
        draft_config(args.file, args.taxonomy, args.company, args.currency, args.out)
        return 0


if __name__ == "__main__":
    sys.exit(main())
