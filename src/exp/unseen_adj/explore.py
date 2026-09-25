import sys, os, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import stress_class, n_vow
from validate import load_splits
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test_features.csv'); dv = pd.read_csv('data/dev.csv')
wug = pd.read_csv('data/wug_seeds.csv')
train_lem = set(tr.lemma_ru); wug_lem = set(wug.lemma_ru)
te_unseen = te[~te.lemma_ru.isin(train_lem) & ~te.lemma_ru.isin(wug_lem)]
print('test unseen rows', len(te_unseen), 'lemmas', te_unseen.lemma_ru.nunique(), te_unseen.pos.value_counts().to_dict())
print('test unseen freq quantiles', te_unseen.drop_duplicates('lemma_ru').lemma_frequency.describe())
adj = tr[tr.pos=='ADJ']
print('train ADJ lemmas', adj.lemma_ru.nunique(), 'freq quantiles', adj.drop_duplicates('lemma_ru').lemma_frequency.describe())
print('test unseen feats dist', te_unseen.feats.value_counts().to_dict())
print('test unseen dialect', te_unseen.dialect.value_counts().to_dict())
print('rows per lemma test unseen', te_unseen.groupby('lemma_ru').size().describe())
print('test unseen yat', te_unseen.drop_duplicates('lemma_ru').yat_flag.mean())
print('train ADJ feats', adj.feats.value_counts().to_dict())
print('train ADJ dialect', adj.dialect.value_counts().to_dict())
# frequency values
print('test unseen freq value counts top', te_unseen.drop_duplicates('lemma_ru').lemma_frequency.value_counts().head(20).to_dict())
print('train ADJ freq value counts top', adj.drop_duplicates('lemma_ru').lemma_frequency.value_counts().head(20).to_dict())
# syllable count
te_unseen_l = te_unseen.drop_duplicates('lemma_ru')
print('test unseen nvow', te_unseen_l.lemma_ru.map(n_vow).value_counts().sort_index().to_dict())
print('train ADJ nvow', adj.drop_duplicates('lemma_ru').lemma_ru.map(n_vow).value_counts().sort_index().to_dict())
tr_rows, known, val = load_splits()
vu = val['unseen']
print('holdout unseen rows', len(vu), 'lemmas', vu.lemma_ru.nunique(), 'freq', vu.drop_duplicates('lemma_ru').lemma_frequency.describe())
print('holdout unseen rows/lemma', vu.groupby('lemma_ru').size().describe())
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl','rb'))
print(ext.keys(), [len(m) for m in ext['unseen_adj']], len(vu))
print(ext['unseen_adj'][0][0])
# class by freq bucket for ADJ cells
adj = adj.copy(); adj['cls'] = adj.form_vvz.map(stress_class)
adj['fb'] = pd.cut(adj.lemma_frequency, [0,100,200,400,1000,3000,1e9], labels=['<100','100-200','200-400','400-1k','1k-3k','>3k'])
for f in ['ADJ;NOM;SG;SHORT','ADJ;NOM;SG;LONG','ADJ;GEN;PL;LONG','ADJ;NOM;PL;SHORT']:
    d = adj[adj.feats==f]
    print(f, pd.crosstab(d.fb, d.cls, normalize='index').round(3).to_dict('index'))
