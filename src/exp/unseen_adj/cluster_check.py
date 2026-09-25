import sys, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ACC, n_vow, stress_class
from joint import combine_tally
from validate import load_splits
strip = lambda s: s.replace(ACC, '')
tr = pd.read_csv('data/train.csv'); adj = tr[tr.pos == 'ADJ']
VOW = set('аеиоуыэюяё')
def cluster_kij(l): return l.endswith('кий') and len(l) > 3 and l[-4] not in VOW  # e.g. гибкий, близкий
lem = adj.drop_duplicates('lemma_ru'); ck = lem[lem.lemma_ru.map(cluster_kij)]
print('train cluster-кий lemmas:', len(ck), list(ck.lemma_ru.head(15)))
d = adj[adj.lemma_ru.isin(set(ck.lemma_ru)) & (adj.feats == 'ADJ;NOM;SG;SHORT')]
for r in d.drop_duplicates('lemma_ru').head(10).itertuples(): print('  ', r.lemma_ru, r.dialect, r.form_vvz)
d2 = adj[adj.lemma_ru.isin(set(ck.lemma_ru)) & (adj.feats == 'ADJ;NOM;SG;LONG')]
for r in d2.drop_duplicates('lemma_ru').head(6).itertuples(): print('  LONG', r.lemma_ru, r.dialect, r.form_vvz)
te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
teu = te[~te.lemma_ru.isin(set(tr.lemma_ru)) & ~te.lemma_ru.isin(set(wug.lemma_ru))].drop_duplicates('lemma_ru')
print('test unseen cluster-кий lemmas:', teu.lemma_ru.map(cluster_kij).sum(), 'of', len(teu), '; nv2 among them', teu[teu.lemma_ru.map(cluster_kij)].lemma_ru.map(n_vow).value_counts().to_dict())
# holdout nv2 lemmas: per-lemma oracle EM and base EM
train_rows, known, val = load_splits(); vu = val['unseen'].reset_index(drop=True)
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl', 'rb'))['unseen_adj']
tallies = [combine_tally(ext, i) for i in range(len(vu))]
gc = vu.form_vvz.map(stress_class).values
def pick(i, c):
    b = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
    return b[0] if b else max(tallies[i].items(), key=lambda x: x[1])[0]
vu['orc'] = [pick(i, gc[i]) == vu.form_vvz[i] for i in range(len(vu))]
vu['nn'] = [max(tallies[i].items(), key=lambda x: x[1])[0] == vu.form_vvz[i] for i in range(len(vu))]
vu['nv'] = vu.lemma_ru.map(n_vow)
print(vu[vu.nv == 2].groupby('lemma_ru')[['orc', 'nn']].mean().round(2).to_string())
print('holdout cluster-кий lemmas:', sorted(set(vu[vu.lemma_ru.map(cluster_kij)].lemma_ru)), vu[vu.lemma_ru.map(cluster_kij)][['orc', 'nn']].mean().round(3).to_dict())
