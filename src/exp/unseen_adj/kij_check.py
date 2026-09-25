import sys, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ACC, n_vow, stress_class
from joint import combine_tally
strip = lambda s: s.replace(ACC, '')
tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv'); full = pd.concat([tr, dv]); adj = full[full.pos == 'ADJ']
VOW = set('аеиоуыэюяё')
nonsk = lambda l: l.endswith('кий') and not l.endswith('ский') and l[-4] not in VOW
lem = adj.drop_duplicates('lemma_ru'); ns = lem[lem.lemma_ru.map(nonsk)]
print('train non-ский cluster-кий lemmas:', len(ns), list(ns.lemma_ru))
for f in ['ADJ;NOM;SG;SHORT', 'ADJ;NOM;SG;LONG', 'ADJ;GEN;SG;SHORT', 'ADJ;NOM;PL;SHORT']:
    d = adj[adj.lemma_ru.isin(set(ns.lemma_ru)) & (adj.feats == f)]
    print(f, [(r.lemma_ru, r.dialect, r.form_vvz) for r in d.head(8).itertuples()])
# also -гий / -хий cluster and vowel+кий (e.g. 'жаркий'->'рк')
te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
mask = (~te.lemma_ru.isin(set(tr.lemma_ru)) & ~te.lemma_ru.isin(set(wug.lemma_ru))).values
teu = te[mask].copy(); teu['i'] = np.where(mask)[0]
tl = teu[teu.lemma_ru.map(nonsk)]
print('test unseen non-ский cluster-кий lemmas:', tl.lemma_ru.nunique(), 'rows', len(tl), list(tl.drop_duplicates('lemma_ru').lemma_ru.head(30)))
ext = pickle.load(open('runs/test_cands_v1v3.pkl', 'rb'))
pat = collections.Counter(); ex = []
for r in tl.itertuples():
    t = combine_tally(ext, r.i); s = max(t.items(), key=lambda x: x[1])[0]
    key = (r.feats.split(';')[1] + ';' + r.feats.split(';')[2] + ';' + r.feats.split(';')[3], strip(s)[-4:])
    pat[key] += 1
    if r.feats in ('ADJ;NOM;SG;SHORT', 'ADJ;NOM;SG;LONG') and len(ex) < 20: ex.append((r.lemma_ru, r.feats, r.dialect, s, round(max(t.values()), 2)))
print('test top-candidate 4-suffix patterns for NOM;SG cells:', {k: v for k, v in pat.items() if k[0].startswith('NOM;SG')})
for e in ex: print('  ', e)
