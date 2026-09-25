import sys, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import stress_class, n_vow, NC
from joint import combine_tally, build_masses
from validate import load_splits
train_rows, known, val = load_splits(); vu = val['unseen'].reset_index(drop=True)
P = pickle.load(open('runs/exp/unseen_adj/suffix_preds_k3_m3.pkl', 'rb'))
base = P['base']['unseen']; nvp = P['nvprior']['unseen']; gold = list(vu.form_vvz)
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl', 'rb'))['unseen_adj']
tallies = [combine_tally(ext, i) for i in range(len(vu))]; masses = build_masses(tallies)
adj = train_rows[train_rows.pos == 'ADJ'].copy(); adj['cls'] = adj.form_vvz.map(stress_class); adj = adj[adj.cls >= 0]
g = adj.groupby(['lemma_ru', 'feats']).cls.agg(lambda s: collections.Counter(s).most_common(1)[0][0]).reset_index(); g['nv'] = g.lemma_ru.map(n_vow).clip(2, 6)
t = g.groupby(['feats', 'nv', 'cls']).size().unstack(fill_value=0).reindex(columns=range(NC), fill_value=0) + 1.0; Pnv = (t.T / t.sum(axis=1)).T
def pick(i, c):
    b = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
    return b[0] if b else max(tallies[i].items(), key=lambda x: x[1])[0]
nvs = vu.lemma_ru.map(n_vow).clip(2, 6).values
outs = {'base(cached)': base, 'nvprior lam2(cached)': nvp}
for lam in (5.0, 1000.0):
    outs[f'nvprior lam{lam}'] = [pick(i, int((np.log(masses[i]) + lam * np.log(Pnv.loc[(vu.feats[i], nvs[i])].values + 1e-9)).argmax())) for i in range(len(vu))]
outs['prior_only(argmax table)'] = [pick(i, int(Pnv.loc[(vu.feats[i], nvs[i])].values.argmax())) for i in range(len(vu))]
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
teu = te[~te.lemma_ru.isin(set(tr.lemma_ru)) & ~te.lemma_ru.isin(set(wug.lemma_ru))].drop_duplicates('lemma_ru')
tnv = teu.lemma_ru.map(n_vow).clip(2, 6).value_counts(normalize=True); hnv = vu.drop_duplicates('lemma_ru').lemma_ru.map(n_vow).clip(2, 6).value_counts(normalize=True)
w = np.array([tnv.get(v, 0) / hnv.get(v, 1e-9) for v in nvs])
for k, o in outs.items():
    e = np.array([a == b for a, b in zip(o, gold)], float)
    print(f'{k:26s} EM {e.mean():.4f}  test-nv-reweighted {np.average(e, weights=w):.4f}  n={len(e)}')
