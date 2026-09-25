"""Label-free diagnostic on test candidates: how often is the top candidate's ending invalid per train (k=3, min_count=3)?"""
import sys, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ACC, stress_class
from joint import combine_tally
strip = lambda s: s.replace(ACC, '')
tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv'); te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
full = pd.concat([tr, dv])
suf = collections.defaultdict(collections.Counter)
for p_, f, d, fm in zip(full.pos, full.feats, full.dialect, full.form_vvz): suf[(p_, f, d)][strip(fm)[-3:]] += 1
valid = lambda p_, f, d, s: suf[(p_, f, d)][strip(s)[-3:]] >= 3
ext = pickle.load(open('runs/test_cands_v1v3.pkl', 'rb')); print('models', len(ext), 'rows', len(ext[0]))
trl = set(tr.lemma_ru); wl = set(wug.lemma_ru); lf = set(zip(tr.lemma_ru, tr.feats))
seg = np.where(te.lemma_ru.isin(trl), np.where([(l, f) in lf for l, f in zip(te.lemma_ru, te.feats)], 'transfer', 'complete'), np.where(te.lemma_ru.isin(wl), 'wug', 'unseen'))
te['seg'] = seg; print(te.seg.value_counts().to_dict())
top = []; inv = []; any_valid = []; repairable = []
for i, r in enumerate(te.itertuples()):
    t = combine_tally(ext, i); s = max(t.items(), key=lambda x: x[1])[0]; top.append(s)
    v = valid(r.pos, r.feats, r.dialect, s); inv.append(not v)
    any_valid.append(any(valid(r.pos, r.feats, r.dialect, c) for c in t))
    repairable.append((not v) and r.pos == 'ADJ' and r.feats.endswith('LONG') and strip(s).endswith('ии') and not strip(s).endswith('иии') and valid(r.pos, r.feats, r.dialect, s + 'и'))
te['inv'] = inv; te['anyv'] = any_valid; te['rep'] = repairable
print('top-candidate invalid rate by seg:', te.groupby('seg').inv.mean().round(4).to_dict())
print('rows with top invalid but some valid cand in beam, by seg:', te[te.inv].groupby('seg').anyv.mean().round(3).to_dict())
print('repairable (append и) by seg:', te.groupby('seg').rep.sum().to_dict())
x = te[te.inv].groupby(['seg', 'feats']).size().sort_values(ascending=False).head(15); print(x.to_string())
u = te[te.seg == 'unseen']; print('unseen NOM;PL;LONG rows', (u.feats == 'ADJ;NOM;PL;LONG').sum(), 'invalid among them', u[u.feats == 'ADJ;NOM;PL;LONG'].inv.sum())
w = te[te.seg == 'wug']; print('wug invalid top by feats', w[w.inv].feats.value_counts().head(8).to_dict())
for r in te[te.inv & (te.seg == 'wug')].head(8).itertuples(): print('  wug invalid ex', r.lemma_ru, r.feats, r.dialect, top[r.Index], dict(suf[(r.pos, r.feats, r.dialect)].most_common(3)))
