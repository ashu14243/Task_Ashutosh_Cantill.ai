"""Per-category zero-label report (dev helper): python scripts/zs_report.py A|B"""
import sys
import pandas as pd
c = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 50); pd.set_option('display.max_rows', 200)
o = pd.read_csv(f'outputs/eval/{c}_zero_label.csv', dtype=str, keep_default_na=False)
if c == 'A':
    l = pd.read_csv('data/company_A/labels_A.csv', dtype=str, keep_default_na=False)
    e = pd.read_csv('data/company_A/erp_goods_receipts.csv', dtype=str, keep_default_na=False)
    e['record_id'] = 'ERP:' + e.line_id; e['text'] = e.material_short_text
    m = o.merge(l, on='record_id').merge(e[['record_id', 'text']], on='record_id')
else:
    l = pd.read_csv('data/company_B/seed_labels.csv', dtype=str, keep_default_na=False).rename(columns={'line_id': 'record_id'})
    e = pd.read_csv('data/company_B/invoice_lines.csv', dtype=str, keep_default_na=False).rename(columns={'line_id': 'record_id'})
    e['text'] = e.line_description + ' || ' + e.gl_account_description
    m = o.merge(l, on='record_id').merge(e[['record_id', 'text']], on='record_id')
    m = m[m.include_y == '1']
m['spend'] = m.spend.astype(float).abs(); m['ok'] = m.l3 == m.category_l3; m['ok1'] = m.l1 == m.category_l1
g = m.groupby('category_l3').agg(n=('ok', 'size'), acc=('ok', 'mean'), acc1=('ok1', 'mean'), spend=('spend', 'sum')).sort_values('spend', ascending=False)
g['spend_acc'] = m.groupby('category_l3').apply(lambda x: (x.ok * x.spend).sum() / max(x.spend.sum(), 1e-9))
print(g.round(3).to_string())
if len(sys.argv) > 2:
    x = m[~m.ok].groupby(['text', 'category_l3', 'l3']).agg(spend=('spend', 'sum'), n=('ok', 'size'), conf=('confidence', 'first')).sort_values('spend', ascending=False)
    print(x.head(int(sys.argv[2])).to_string())
