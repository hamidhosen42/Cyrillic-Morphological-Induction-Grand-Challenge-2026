"""Leave-lemma-out CV of the class prior on train ADJ lemmas (no NN): global vs nvow / freq conditioned."""
import sys, os, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import stress_class, n_vow, NC
tr = pd.read_csv('data/train.csv')
adj = tr[tr.pos == 'ADJ'].copy(); adj['cls'] = adj.form_vvz.map(stress_class); adj = adj[adj.cls >= 0]
# majority class per (lemma, feats)
g = adj.groupby(['lemma_ru', 'feats']).agg(cls=('cls', lambda s: collections.Counter(s).most_common(1)[0][0]),
                                           freq=('lemma_frequency', 'first')).reset_index()
g['nv'] = g.lemma_ru.map(n_vow).clip(2, 6)
g['fb'] = pd.cut(g.freq, [0, 300, 500, 1000, 3000, 1e9], labels=False)
g['fb2'] = pd.cut(g.freq, [0, 400, 1000, 1e9], labels=False)
print('rows', len(g), 'lemmas', g.lemma_ru.nunique())
print('nv dist (lemmas)', g.drop_duplicates('lemma_ru').nv.value_counts().sort_index().to_dict())
for c in ['ADJ;NOM;SG;SHORT', 'ADJ;NOM;SG;LONG', 'ADJ;GEN;PL;LONG', 'ADJ;INS;SG;SHORT']:
    d = g[g.feats == c]
    print(c, 'by nv:', pd.crosstab(d.nv, d.cls, normalize='index').round(2).to_dict('index'))
# LOO CV of argmax prior with 5-fold lemma folds
lem = np.array(sorted(g.lemma_ru.unique())); rng = np.random.default_rng(0); rng.shuffle(lem)
fold = {l: i % 5 for i, l in enumerate(lem)}
g['fold'] = g.lemma_ru.map(fold)
def cv(keys, alpha=1.0):
    acc = np.zeros(len(g)); ll = np.zeros(len(g))
    for f in range(5):
        trn = g[g.fold != f]; tst = g[g.fold == f]
        tab = trn.groupby(keys + ['cls']).size().unstack(fill_value=0)
        tab = tab.reindex(columns=range(NC), fill_value=0) + alpha
        glob = trn.groupby(['feats', 'cls']).size().unstack(fill_value=0).reindex(columns=range(NC), fill_value=0) + alpha
        glob = glob.div(glob.sum(1), axis=0)
        P = tab.div(tab.sum(1), axis=0)
        for i, r in zip(tst.index, tst.itertuples()):
            k = tuple(getattr(r, x) for x in keys)
            if k in P.index: p = P.loc[k].values
            else: p = glob.loc[r.feats].values
            acc[i] = p.argmax() == r.cls; ll[i] = np.log(p[r.cls])
    return acc, ll
res = {}
for name, keys in [('global', ['feats']), ('nv', ['feats', 'nv']), ('fb', ['feats', 'fb']), ('fb2', ['feats', 'fb2']),
                   ('nv+fb2', ['feats', 'nv', 'fb2']), ('nv+fb', ['feats', 'nv', 'fb'])]:
    acc, ll = cv(keys); res[name] = acc
    print(f'{name:8s} class acc {acc.mean():.4f}  mean loglik {ll.mean():.4f}')
# test-like reweighting: test unseen nv dist
te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
teu = te[~te.lemma_ru.isin(set(tr.lemma_ru)) & ~te.lemma_ru.isin(set(wug.lemma_ru))].drop_duplicates('lemma_ru')
tnv = teu.lemma_ru.map(n_vow).clip(2, 6).value_counts(normalize=True).sort_index()
hnv = g.drop_duplicates('lemma_ru').nv.value_counts(normalize=True).sort_index()
w = g.nv.map(lambda v: tnv.get(v, 0) / hnv.get(v, 1e-9))
print('test nv dist', tnv.round(3).to_dict())
for name, acc in res.items():
    print(f'{name:8s} test-nv-reweighted class acc {np.average(acc, weights=w):.4f}   ' +
          ' '.join(f'nv{v}:{acc[g.nv == v].mean():.3f}' for v in range(2, 7)))
