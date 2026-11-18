# Decisions

What the data showed and what the tool does about it. Counts are from the build data
(`python -m spendtool run ...` summaries and diagnostics). Everything here is a config value or a
generic rule; nothing is hard-coded per company.

## Load-bearing assumptions (if one is wrong, the listed numbers move)

1. **All of A's ERP goods-receipt lines are spend.** The ERP file has no status column and the labels
   give every one of its 37,029 lines a category. Negative receipts (434, returns) stay as negative
   spend; 509 lines of ZAR 0.01 stay in.
2. **A tool line and an ERP receipt with the same PO and the same net amount (± max(ZAR 1, 0.2%)) are
   the same purchase**, and one ERP receipt absorbs at most one tool line (plus split-invoice parts
   and exact re-exports). A second tool invoice for the same amount under a *different* invoice
   number is treated as separate, tool-only spend.
3. **A status reason beats DUPLICATE.** A cancelled tool line that also matches an ERP receipt is
   reported as CANCELLED (the ERP line is still counted, so spend is unaffected).
4. **B's line amounts are net.** Tax is booked on its own lines (e.g. "Sales Tax Expense"), which are
   excluded as NON_PROCUREMENT.
5. **For A2/B2 the original A/B files stay on disk**: the configs load them as labelled training rows
   when a run points at new files (`labels.training_files`). Their rows never reach the output.

## Company A

**Dates.** Tool `invoice_date` has three formats (`2024-03-29`, `16/05/2024`, `27-Mar-24`).
2,050 slash dates have a first part > 12 and none a second part > 12, so slash dates are day-first.
18,660 tool dates and 24,370 ERP posting dates are blank: flagged `BLANK` in the diagnostics; no
spend decision depends on a date (dates only break ties in matching).

**VAT.** `vat_rate` comes as `15`, `0.15`, `15%`, `0` or blank. Blank (6,542) → 15% per the brief.
`0` (1,580) is taken literally as 0%: e.g. TA004389, 360 units at the ERP unit cost of ZAR 453.56
= 163,281.60 against a "gross" of 163,280.56, i.e. the amount is already net. Amounts with thousands
commas (`"5,266.63"`) are parsed.

**Status.** Paid (19,837) and Approved (8,180) count; Cancelled (188) → CANCELLED; On Hold (172) and
Rejected (143) → NOT_APPROVED.

**PO references** come as `4510062123`, `PO 4510062123`, `PO#4510062123`, `4510062123/10.0` or blank
(5,199). The config extracts the 10-digit PO and, when present, the PO item after the slash, which
must then equal the ERP `po_item`.

**Count once (tool vs ERP).** Of 28,017 status-eligible tool lines:

| Rule | Lines | Note |
|---|---|---|
| same PO, net amount within tolerance | 16,691 | PO item must agree when both have one |
| same supplier + amount + text ≥ 0.6 (PO blank or unknown to the ERP) | 5,743 | 93% also have identical quantity; amounts match to the cent |
| split invoice: 2-4 lines of one invoice sum to one receipt | 28 | e.g. TAX INV 175122: 6,500 + 5,500 units = receipt A011328 (12,000) |
| same tool line exported twice | 650 | same supplier, invoice digits and amount; PO item must not differ |
| **tool-only spend kept** | **4,905** | ZAR 359.2m |

Lines on a known PO whose amount matches no receipt (e.g. PO 4510019355: ERP receipt of 5 units,
tool invoices of 165 and 360 units at the same unit price) are kept as tool-only spend, not
duplicates. Consistency check (not proof): ERP line IDs run A000002-A045000 with 7,970 gaps; at
the tool's ~59% coverage of receipts that predicts roughly 4,700 tool-only lines; 4,905 were found.
Matching is weak-evidence-proof: text similarity alone never merges records, an amount match is
always required.

**OCR noise** in tool text and names (`P0WDER`, `RE5IN`, `8OLT`, `KEL MONT`, `OAK ENT MINING`): digits
inside alphabetic tokens are mapped back to letters, spaces are removed in supplier cores.

**Suppliers.** 193 ERP names and 702 raw tool spellings normalise to 198 distinct supplier cores;
every tool spelling lands on an ERP name once legal forms are stripped only at the end of a name (so
`AS HAX` stays `ASHAX`) and OCR/space noise is removed. 198 cores → 77 keys with labels (198 without). The golden key SG-f85711 groups 122 unrelated-looking names (20,891 ERP lines); with labels we
reproduce it, without labels nothing name-based can.

**Items.** The golden item key is the material number (each material maps to exactly one key); lines
without a material (services, free text: 15,816) have no golden item. Tool lines take the item of the
ERP line they duplicate, or of the catalogue text they match (same numbers required). Uninformative
texts ("No Object" ×914, "Mechanical Maintenan") are keyed per supplier.

## Company B

**Exclusions.** No status column, so only NON_PROCUREMENT applies. All 450 excluded seeds are
intercompany (GL descriptions `IC06`, `IC05`, `IC22`..., `IC Payables`, `I/C Payable`) or tax lines
(`State & Local Taxes`, `Sales Tax Expense`, `GST Input Tax`, `Sales Tax`). The generic rules
reproduce all 3,000 seed decisions with the seeds hidden. Full file: 2,489 excluded (1,068 by
intercompany text, 973 by IC GL code, 448 tax). Kept as spend on purpose: customs/import duties (the
seeds label them *Customs Clearance & Duties*), "Assistance with Property Tax" (advice) and "Payroll
Services" (an outsourced service). Rules learned from labels need ≥ 10 labelled lines at ≥ 98%
excluded; with the full seeds they add nothing beyond the generic rules.

**Duplicates.** 1,289 lines repeat system + invoice + line + amount + vendor (mostly JDE ledger
repeats). Eight of them are in the seeds: six labelled as spend, two as intercompany, none as
DUPLICATE, so B has no line de-dup.

**Currency.** 17,959 USD lines; 1,279 of them carry a rate around 17 — ignored, USD is already USD.
Inline rates agree with the fx table (median ratio per currency between 0.98 and 1.01) and are used as given:
USD = amount / rate. 3,343 non-USD lines without a rate use the table for the invoice month (3,338),
else the posting month, else the nearest month (5). Four invoice dates are implausible (2018, 2027):
flagged, posting month used. Three lines have a blank amount (currency RCN): spend 0, flagged.

**Text.** 16,695 lines (41%) have a blank description, concentrated in ERP-B1/B6/B8; they are
classified from GL text plus their supplier's and item's other lines. `MAT-1000007` is a catch-all
material (10,484 lines, many unrelated texts) and is not used as an item key. Suppliers are vendor IDs
only, so `supplier_key` = vendor ID: the same real supplier under two IDs in two ERPs cannot be
detected without names. Seeds are noisy (e.g. "Microsoft Server Cloud Enrollment" labelled as
hardware), so they are blended with the tree-based evidence rather than trusted absolutely.

## Classification choices

* Lexical zero-shot was chosen over a hosted model so the tool runs offline, deterministically, in
  minutes; the LLM step is an optional layer, not a dependency.
* Flat arg-max over leaves beat top-down decoding on both A and B with labels hidden; length
  normalisation 0.75 beat plain cosine (B L3 by spend 23.6% → 29.5%, A 40.5% → 40.7%).
* Excluded rows carry no category (as in `output_template.csv`); their `confidence` is the
  confidence of the exclusion (1.0 status, 0.95 PO+amount match, 0.9 rule, 0.8 other matches).

## Not verified

* How the blind test scores grouping and confidence is not stated; keys are optimised for precision.
* The LLM step was tested only against a local fake server (no API key was available).
* Company C's format is unknown; `profile` / `init-config` draft its config, which must be reviewed.
