import sys, os, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ParadigmModel, stress_class, n_vow, NC
from joint import combine_tally, build_masses
from validate import load_splits
train_rows, known, val = load_splits()
vu = val['unseen'].reset_index(drop=True)
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl', 'rb'))['unseen_adj']
tallies = [combine_tally(ext, i) for i in range(len(vu))]; masses = build_masses(tallies)
gold = list(vu.form_vvz); gcls = np.array([stress_class(f) for f in gold])
pms = [ParadigmModel(K=300, seed=s).fit(train_rows) for s in (0, 1, 2)]
prior = {f: sum(pm.class_dist('ADJ', f, []) for pm in pms) / 3 for f in vu.feats.unique()}
def pick(i, c):
    best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
    return best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]
# per-row decision (test-like: groups mostly singletons)
pred = []; pcls = []
for i in range(len(vu)):
    sc = np.log(masses[i]) + 2.0 * np.log(prior[vu.feats[i]] + 1e-9); c = int(sc.argmax()); pcls.append(c); pred.append(pick(i, c))
orc = [pick(i, gcls[i]) if gcls[i] >= 0 else pred[i] for i in range(len(vu))]
nn = [max(tallies[i].items(), key=lambda x: x[1])[0] for i in range(len(vu))]
vu['em'] = [a == b for a, b in zip(pred, gold)]; vu['orc'] = [a == b for a, b in zip(orc, gold)]
vu['nn'] = [a == b for a, b in zip(nn, gold)]; vu['clsok'] = np.array(pcls) == gcls
vu['top1_in_beam'] = [gold[i] in tallies[i] for i in range(len(vu))]
print('holdout dialect rows', vu.dialect.value_counts().to_dict())
print(vu.groupby('dialect')[['em', 'orc', 'nn', 'clsok', 'top1_in_beam']].mean().round(4))
print(vu.groupby('yat_flag')[['em', 'orc', 'nn', 'clsok']].mean().round(4))
vu['ls'] = vu.feats.str.split(';').str[-1]
print(vu.groupby('ls')[['em', 'orc', 'clsok']].mean().round(4))
# test unseen dialect / rows composition
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
teu = te[~te.lemma_ru.isin(set(tr.lemma_ru)) & ~te.lemma_ru.isin(set(wug.lemma_ru))]
td = teu.dialect.value_counts(normalize=True); print('test unseen dialect frac', td.round(3).to_dict())
hd = vu.dialect.value_counts(normalize=True)
w = vu.dialect.map(lambda d: td[d] / hd[d])
print('dialect-reweighted holdout: em', np.average(vu.em, weights=w).round(4), 'orc', np.average(vu.orc, weights=w).round(4))
# yat frac
print('test unseen yat frac (lemmas)', teu.drop_duplicates('lemma_ru').yat_flag.mean(), 'holdout', vu.drop_duplicates('lemma_ru').yat_flag.mean(), 'train ADJ', tr[tr.pos=='ADJ'].drop_duplicates('lemma_ru').yat_flag.mean())
# 2-syllable test lemmas
l2 = teu.drop_duplicates('lemma_ru'); l2 = l2[l2.lemma_ru.map(n_vow) == 2]
print('test nv2 lemma sample', list(l2.lemma_ru.head(40)))
print('test nv2 endings', l2.lemma_ru.str[-2:].value_counts().to_dict())
adj = tr[tr.pos == 'ADJ'].drop_duplicates('lemma_ru')
print('train ADJ endings', adj.lemma_ru.str[-2:].value_counts().head(8).to_dict())
print('train nv2 endings', adj[adj.lemma_ru.map(n_vow) == 2].lemma_ru.str[-2:].value_counts().to_dict())
print('test unseen endings', teu.drop_duplicates('lemma_ru').lemma_ru.str[-2:].value_counts().head(8).to_dict())
# KAM errors: examples
kam = vu[(vu.dialect == 'KAM') & (~vu.orc)]
for r in kam.head(12).itertuples(): print('  KAM orc-miss', r.lemma_ru, r.feats, 'gold', r.form_vvz, 'pred', orc[r.Index])
sev = vu[(vu.dialect == 'SEV') & (~vu.orc)]
for r in sev.head(8).itertuples(): print('  SEV orc-miss', r.lemma_ru, r.feats, 'gold', r.form_vvz, 'pred', orc[r.Index])
