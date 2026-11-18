"""Dev helper: zero-label accuracy under config variations.  python scripts/ablate.py A 'classification.self_training=false' ..."""
import copy, json, sys, logging
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from spendtool.config import load_config
from spendtool.classify.taxonomy import load_taxonomy
from spendtool.pipeline import load_labels, run
from spendtool.evaluate import _accuracy
logging.basicConfig(level=logging.WARNING)
c = sys.argv[1]
cfg = load_config(f'configs/company_{c}.yaml'); tax = load_taxonomy(cfg); lab = load_labels(cfg, tax)
paths = np.full((len(lab), 4), '', dtype=object)
for i, lf in enumerate(lab['_leaf']):
    if pd.notna(lf):
        p = tax.leaves[int(lf)]; paths[i, :len(p)] = p
gold = pd.DataFrame(paths, index=lab.index, columns=['l1', 'l2', 'l3', 'l4'])
for variant in ['base'] + sys.argv[2:]:
    cc = copy.deepcopy(cfg); cc['output']['diagnostics'] = False
    if variant != 'base':
        for kv in variant.split(';'):
            k, v = kv.split('='); d = cc; ks = k.split('.')
            for kk in ks[:-1]: d = d[kk]
            d[ks[-1]] = json.loads(v)
    res = run(cc, use_labels=False, output=f'outputs/eval/{c}_ablate.csv')
    out = res['frame'].set_index('record_id')
    rows = lab.index[lab['_leaf'].notna()].intersection(out.index[out['include'] == 1])
    acc = _accuracy(out, gold, rows, tax.max_depth)
    print(f"{c} {variant:60s}", {k: (v['by_count'], v['by_spend']) for k, v in acc.items()}, flush=True)
