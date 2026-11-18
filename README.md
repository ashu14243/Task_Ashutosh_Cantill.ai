# Cantill spend tool

A config-driven pipeline that turns a company's raw purchase records and its category tree into a
spend cube: every input record gets one output row saying whether it counts as spend (and why
not), its net spend in the reporting currency, a stable supplier key, a stable item key, and a
category path from the company's own tree with a confidence.

Nothing about a company lives in the code. A company is a YAML file (`configs/company_A.yaml`,
`configs/company_B.yaml`); a new company needs a new YAML file only. Category names are read from
the taxonomy file at run time.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                           # Python 3.10+
```

Put the task kit's data folders under `data/` (they are gitignored and never committed):

```
data/company_A/erp_goods_receipts.csv, invoice_tool_export.csv, labels_A.csv, taxonomy_A.csv
data/company_B/invoice_lines.csv, fx_rates_usd.csv, seed_labels.csv, taxonomy_B.csv
```

(Or keep them anywhere and add `--data-dir /path/to/kit` to every command.)

## The one command

```bash
python -m spendtool run --config configs/company_A.yaml     # -> outputs/ashutosh__A.csv
python -m spendtool run --config configs/company_B.yaml     # -> outputs/ashutosh__B.csv
```

Each run also prints a summary, writes `<output>.summary.json` (counts by exclusion reason, total
spend, date/FX flags, runtime) and `<output>.diagnostics.csv` (per-row audit trail: rule that
excluded it, which ERP line a tool line duplicates, VAT rate and FX rate used, parsed date and date
flag, classifier source). Both side files stay local (gitignored). Every run validates its own
output (one row per input, valid reason codes, zero spend on excluded rows, every category path
exists in the tree) and logs any problem.

Runtime on a laptop-class 2-core machine: A (65,549 rows) ~100 s, B (41,028 rows) ~55 s. Runs are
deterministic: the same inputs give the same output.

## Blind test runbook (19 Nov, 10:00-12:00 IST)

Keep `data/company_A` and `data/company_B` in place: the A/B configs read the original labelled
files as training data when they are pointed at new files.

```bash
# A2 and B2: existing configs, new files (paths are examples - use the names that arrive)
python -m spendtool run --config configs/company_A.yaml \
    --input erp=data/company_A2/erp_goods_receipts.csv --input tool=data/company_A2/invoice_tool_export.csv \
    --output submission/ashutosh__A2.csv
python -m spendtool run --config configs/company_B.yaml \
    --input invoices=data/company_B2/invoice_lines.csv --output submission/ashutosh__B2.csv
#   (if B2 comes with its own fx_rates_usd.csv, copy it over data/company_B/fx_rates_usd.csv first
#    or point fx.file at it in a copy of the config - configs are data, not code)

# C: look at the file, draft a config, review it, run
python -m spendtool profile --file data/company_C/<invoices>.csv
python -m spendtool init-config --file data/company_C/<invoices>.csv --taxonomy data/company_C/<tree>.csv \
    --company C --currency USD --out submission/company_C.yaml
#   edit submission/company_C.yaml: amount column + net/gross + default VAT, currency/fx, status map,
#   non-procurement rules for the company's GL/intercompany codes, duplicates rule, dayfirst
python -m spendtool run --config submission/company_C.yaml --output submission/ashutosh__C.csv

git add -f submission/ashutosh__A2.csv submission/ashutosh__B2.csv submission/ashutosh__C.csv submission/company_C.yaml
git commit -m "Blind test outputs and company C config" && git push
```

Checklist for the C config: (1) every status value is mapped (the draft lists them all);
(2) amount is net, or `amount_type: gross` with the right tax column/default rate; (3) currency and
FX direction; (4) look at the `profile` output for GL/account descriptions naming intercompany,
taxes or other non-spend and add `non_procurement.rules`; (5) if the file can contain the same
invoice line twice, add a `duplicates.within_source` rule; (6) check `rows=` and `excluded=` in the
run summary look sensible.

## Optional LLM (off unless a key is present)

The tool runs fully offline. If `ANTHROPIC_API_KEY` is set (or `classification.llm` points at any
OpenAI-compatible endpoint, including a free local Ollama server), the highest-spend distinct lines
are also classified by an LLM and blended with the local classifiers. Failures, timeouts or a bad key
are logged and the run continues without it; `--no-llm` forces it off. Cost estimate in
[APPROACH.md](APPROACH.md).

```bash
export ANTHROPIC_API_KEY=...            # or: SPENDTOOL_LLM_API_KEY / OPENAI_API_KEY
python -m spendtool run --config submission/company_C.yaml --output submission/ashutosh__C.csv
```

## Other commands

```bash
python -m spendtool evaluate --config configs/company_A.yaml --with-labels   # accuracy with labels hidden + 20% holdout
python -m spendtool run --config configs/company_A.yaml --no-labels           # zero-label run
python -m pytest -q                                                           # 50 unit / end-to-end tests
```

## Layout

```
spendtool/
  config.py          load + validate YAML/JSON configs, defaults
  ingest.py          read sources, map columns to canonical fields, build record IDs
  normalize.py       amount / rate / date parsing, OCR repair, supplier name normalisation
  valuation.py       net of tax, FX to reporting currency, date flags
  spend_rules.py     status -> CANCELLED / NOT_APPROVED, NON_PROCUREMENT rules
  dedup.py           supplier keys and item keys (union-find over exact, fuzzy and labelled links)
  matching.py        cross-system matching (count once) and repeated exports
  classify/          taxonomy loader, generic lexicon, zero-shot + supervised + self-training + LLM
  pipeline.py        orchestration, output, validation, summary
  evaluate.py        accuracy and B-cubed key quality against labels
  profiling.py       profile an unseen file, draft its config
configs/             company_A.yaml, company_B.yaml, TEMPLATE.yaml (every option)
outputs/             ashutosh__A.csv, ashutosh__B.csv
tests/               parsers, rules, end-to-end on a synthetic unseen company, LLM against a fake server
```

See [APPROACH.md](APPROACH.md) for the method and measured accuracy and
[DECISIONS.md](DECISIONS.md) for what was found in the data and decided.
