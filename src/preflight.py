"""Submission preflight: schema/row/ID/NaN/accent checks + change breakdown vs a reference submission."""
import sys, collections, unicodedata
import numpy as np, pandas as pd
new, ref = sys.argv[1], sys.argv[2]
te = pd.read_csv('data/test_features.csv'); ss = pd.read_csv('data/sample_submission.csv')
tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv'); wug = pd.read_csv('data/wug_seeds.csv')
a = pd.read_csv(new); b = pd.read_csv(ref)
ok = True
def chk(cond, msg):
    global ok
    print(('OK   ' if cond else 'FAIL ') + msg); ok &= bool(cond)
chk(list(a.columns) == list(ss.columns), f'columns {list(a.columns)}')
chk(len(a) == len(ss) == 60000, f'rows {len(a)}')
chk(a.id.is_unique and (a.id.values == ss.id.values).all(), 'ids unique and in sample order')
chk(a.form_vvz.notna().all() and (a.form_vvz.str.len() > 0).all(), 'no empty/NaN predictions')
chk((a.form_vvz.str.strip() == a.form_vvz).all(), 'no leading/trailing whitespace')
chk((a.form_vvz.map(lambda s: unicodedata.normalize('NFC', s) == s or unicodedata.normalize('NFD', s) == s)).all(), 'unicode normalization consistent')
acc = a.form_vvz.str.count('́')
chk((acc == 1).mean() > 0.999, f'exactly one accent: {(acc == 1).mean():.4f}')
lf = set(zip(pd.concat([tr, dv]).lemma_ru, pd.concat([tr, dv]).feats)); trl = set(tr.lemma_ru); wl = set(wug.lemma_ru)
seg = np.array(['transfer' if (l, f) in lf else 'complete' if l in trl else 'wug' if l in wl else 'unseen' for l, f in zip(te.lemma_ru, te.feats)])
ch = (a.form_vvz.values != b.form_vvz.values)
plain_same = a.form_vvz.str.replace('́', '').values == b.form_vvz.str.replace('́', '').values
print(f'changed vs reference: {ch.sum()} ({ch.mean():.2%}); stress-only changes: {(ch & plain_same).sum()}')
for s in ['transfer', 'complete', 'wug', 'unseen']:
    m = seg == s
    print(f'   {s:9s} n={m.sum():6d} changed {ch[m].sum():6d} ({ch[m].mean():.2%})  stress-only {(ch & plain_same)[m].sum()}')
print('PREFLIGHT', 'PASSED' if ok else 'FAILED')
