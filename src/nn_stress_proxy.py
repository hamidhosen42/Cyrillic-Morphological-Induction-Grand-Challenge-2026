"""Label-free proxy for NN stress-evidence quality on the test: agreement of the NN-only stress class (cur5)
with the paradigm prior's argmax on 'complete' rows (prior ~95% accurate there). Compares tallies built from
the original beams vs rescored pools."""
import sys, pickle, collections, os
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import stress_class, ParadigmModel, NC
te = pd.read_csv('data/test_features.csv'); tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv'); wug = pd.read_csv('data/wug_seeds.csv')
C = pickle.load(open('runs/test_cands_v1v3.pkl', 'rb'))
files = sys.argv[1:]
R = [pickle.load(open(f, 'rb')) for f in files]
al = pd.concat([tr, dv]); lf = set(zip(al.lemma_ru, al.feats)); trl = set(al.lemma_ru)
comp = [i for i, (l, f) in enumerate(zip(te.lemma_ru, te.feats)) if l in trl and (l, f) not in lf]
pms = [pickle.load(open(f'runs/final/pm_cur5_K600_s{s}_full.pkl', 'rb')) for s in range(3)]
km = collections.defaultdict(list)
for l, f, fm in zip(al.lemma_ru, al.feats, al.form_vvz): km[l].append((f, fm))
def beam_mass(i):
    m = np.full(NC, 1e-12)
    for mdl in C:
        lp = np.array([c[1] for c in mdl[i]]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
        for (s, _), q in zip(mdl[i], pr):
            c = stress_class(s)
            if c >= 0: m[c] += q
    return m
def resc_mass(i):
    k = (te.lemma_ru[i], te.feats[i], te.dialect[i]); m = np.full(NC, 1e-12)
    for sc in R:
        d = sc.get(k)
        if not d: continue
        ks = list(d); lp = np.array([d[x] for x in ks]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
        for s, q in zip(ks, pr):
            c = stress_class(s)
            if c >= 0: m[c] += q
    return m
agree_b = agree_r = n = 0; chg = 0
for i in comp:
    r = te.iloc[i]
    pr = sum(pm.class_dist(r.pos, r.feats, km[r.lemma_ru]) for pm in pms)
    cp = int(pr.argmax()); b = int(beam_mass(i).argmax()); rr = int(resc_mass(i).argmax())
    agree_b += (b == cp); agree_r += (rr == cp); chg += (b != rr); n += 1
print(f'complete rows {n}: NN-only stress agrees with prior: beam tally {agree_b / n:.4f} | rescored ({len(R)} models) {agree_r / n:.4f} | argmax changed {chg}')
