# Approach

## Pipeline (one command, config-driven)

`ingest → value → decide → key → count once → classify → validate → write`

1. **Ingest** (`ingest.py`): each source in the YAML maps its columns to ~25 canonical fields
   (amount, tax rate, currency, dates, supplier, description, item code, PO, invoice, status, GL...).
   A field can be a column, several columns, a constant or a regex extract (e.g. the 10-digit PO
   inside `PO#4510062123/10.0`). Record IDs come from a template (`ERP:{id}`).
2. **Value** (`valuation.py`): robust amount parsing (`1,234.56`, `1.234,56`, `(12)`, `12-`), VAT
   removal for gross sources (`15`, `15%`, `0.15`, blank → default), FX to the reporting currency
   (inline rate first; reporting-currency lines are never converted; else the fx table for the
   invoice month, falling back to posting month, then nearest month). Every date format seen is
   parsed; blank, unparseable and implausible dates are flagged in the diagnostics file.
3. **Decide** (`spend_rules.py`): status → `CANCELLED` / `NOT_APPROVED` (config map + generic
   patterns). `NON_PROCUREMENT` from generic rules over description + GL text + supplier
   (intercompany, taxes, financing, royalties/revenue, payroll, depreciation, donations; with
   allow-rules so *tax advice* or *payroll services* stay spend), plus config rules, rules learned
   from labelled exclusions, and supplier propagation.
4. **Key** (`dedup.py`): union-find over base labels. Suppliers: normalised core name (OCR repair,
   legal suffixes stripped only at the end, spaces removed) → exact, then fuzzy ≥ 92 (blocked),
   linked by shared vendor ID, seeded by labelled golden keys when available. Items: material code
   (catch-all codes used for many unrelated texts are ignored) else compact OCR-repaired text; text
   lines are assigned one-way to a catalogue code; fuzzy merges require identical numbers, so
   `DETONATOR 3,5M,7` never merges with `...,8`; generic texts ("No Object", "Parts") are keyed per supplier.
5. **Count once** (`matching.py`): secondary lines (A's tool) are matched to primary lines (A's ERP):
   (a) same PO (+ PO line), net amount within max(1, 0.2%), best quantity/text/date, greedy 1:1;
   (b) no usable PO: same supplier key + amount + quantity + text ≥ 0.6 + date window;
   (c) split invoices: 2-4 lines of one invoice summing to one receipt. Then repeated exports of the
   same tool line (same supplier, invoice digits, amount; compatible PO line; similar text). ERP is
   always the counted copy; the duplicate inherits its supplier/item key.
6. **Classify** (`classify/`): see below. 7. **Validate**: row count, columns, reasons, zero spend on
   exclusions, every path in the tree; summary JSON with totals, flags and runtime.

## Classifying into an unseen tree

The tree is read at run time (level columns or parent/child; up to 4 levels). Evidence is blended as
probability vectors over leaves:

* **Zero-shot lexical** (always on): each leaf is a bag of its path words (leaf 1.0, parent 0.6,
  above 0.35) plus synonyms; each line is a bag of description / GL / supplier / attribute tokens
  plus synonyms (0.7) and broader terms (0.3), after camel-case splitting, OCR repair, abbreviation
  expansion (`MNTN`→maintenance, `VLV`→valve), compound splitting (`ROFBLTR`→roof bolter), light
  lemmatisation to the tree's own vocabulary and acronyms of the tree's names (`CBS`→Can Body Stock).
  The bridge is `lexicon.py`: ~150 generic, multilingual purchasing concepts (EN/ES/DE/FR/IT/PT) —
  general procurement vocabulary, not any company's categories. Score = idf-weighted similarity.
* **Self-training**: a TF-IDF char+word logistic regression fitted on the confident zero-shot
  lines generalises to lines with no shared words (part numbers, house abbreviations).
* **Propagation**: weak lines borrow the spend-weighted distribution of their item and supplier.
* **Labels, when a company has them** (A, B): the same text model trained on labels, item/supplier
  priors from labelled keys, and labelled rows keep their label. For A2/B2 the config points the run
  at the new files and loads the original labelled files alongside as training rows only.
* **Optional LLM** on the highest-spend distinct lines (off unless a key is set).

## Accuracy measured (`python -m spendtool evaluate ... --with-labels`)

| Run | L1 count / spend | L2 count / spend | L3 count / spend |
|---|---|---|---|
| **A, labels hidden (zero-label)** | 63.6% / 78.6% | 38.6% / 55.4% | 27.6% / 40.7% |
| A, 20% of labels hidden | 98.7% / 97.6% | 96.9% / 95.0% | 96.1% / 94.3% |
| A2 simulation (20% of POs held out as new files, `scripts/simulate_a2.py`) | 96.4% / 94.9% | 94.4% / 90.9% | 93.4% / 88.2% |
| B, seeds hidden (zero-label, 2,550 seed lines) | 92.3% / 98.5% | 91.3% / 93.5% | 56.5% / 29.5% |
| B, 20% of seeds hidden | 94.5% / 99.5% | 94.3% / 99.5% | 90.0% / 99.2% |

A's tree is organised by machine (continuous miners, shuttle cars, LHDs) while its lines describe
parts ("MOTOR ELEC 30KW", "GENERAL MECHANICAL"), so the machine-level split is not recoverable from
text alone; L1 by spend is. B's zero-label L3 by spend is dominated by can-body coil lines whose text
is a dimension code (`0124X58.620,25OZ`). Keys with labels hidden, B-cubed against A's golden keys:
items P 1.00 / R 0.88 (F1 0.93); suppliers P 1.00 / R 0.47 — the golden key groups 122 different
names into one supplier, which no name-based method can infer; with labels (A2 simulation) supplier
F1 is 1.00 and item F1 0.98. B's exclusions with seeds hidden agree with the analyst on 100% of the
3,000 seeds.

## Runtime and cost

2-core machine, no GPU: A 65,549 rows in ~95 s, B 41,028 rows in ~55 s, an A2-sized run with
training rows ~125 s — well inside 60 minutes for 25,000 lines. Deterministic (fixed hash seed,
sorted iteration) and identical across pandas 2.2/3.0 and scikit-learn 1.6/1.9.

No paid API was used to produce the outputs here (the LLM step is off without a key). If enabled with
a compatible hosted model, a batch of 40 distinct lines is ~3k input and ~0.6k output tokens, so
1,000 lines cost roughly cents at low-cost published API rates (estimate from token counts, not
measured), and results are cached on disk. AI assistance was used during implementation and
verification through GitHub Copilot using the Copilot SDK in VS Code. No data files or confidential
training inputs were included in the repository or sent to an external model.
