"""Nonparametric paradigm prior: every training lemma is its own paradigm (K = #lemmas), smoothed with
pseudo-count alpha. Weight of training lemma k for a query lemma = prod over observed cells of
P_alpha(k, cell, observed class); prior for target cell = weighted average of P_alpha(k, target, .).
Writes runs/exp/decision-tuning/prior_{seg}_knn{alpha}_lim0.npy (same format as tune.py's caches).
"""
import os, sys, collections, time
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from validate import load_splits
from paradigm import stress_class, NC
OUT = os.path.join(ROOT, 'runs', 'exp', 'decision-tuning')
SEGS = ['transfer', 'complete', 'wug', 'unseen']
ALPHAS = [float(a) for a in (sys.argv[1] if len(sys.argv) > 1 else '0.05,0.15,0.3,0.6').split(',')]

train_rows, known, val = load_splits()
known_map = collections.defaultdict(list)
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map[l].append((f, d, fm))
df = train_rows.copy(); df['cls'] = df.form_vvz.map(stress_class); df = df[df.cls >= 0]
TAB = {}
for pos, d in df.groupby('pos'):
    g = d.groupby(['lemma_ru', 'feats']).cls.agg(lambda s: collections.Counter(s).most_common(1)[0][0])
    x = g.unstack(); cells = list(x.columns); A = x.fillna(-1).values.astype(int)
    TAB[pos] = (cells, {c: i for i, c in enumerate(cells)}, A)
    print(pos, 'lemmas', A.shape[0], 'cells', len(cells), flush=True)

for alpha in ALPHAS:
    P = {}
    for pos, (cells, cidx, A) in TAB.items():
        L, C = A.shape; p = np.full((L, C, NC), 1.0 / NC)
        oh = np.zeros((L, C, NC));
        for c in range(NC): oh[:, :, c] = (A == c)
        obs = (A >= 0)
        p[obs] = (alpha + oh[obs]) / (NC * alpha + 1.0)
        P[pos] = np.log(p)
    for seg in SEGS:
        v = val[seg]; rows = list(v.itertuples()); pr = np.zeros((len(rows), NC)); t = time.time()
        wcache = {}; cache = {}
        for i, r in enumerate(rows):
            cells, cidx, A = TAB[r.pos]; LP = P[r.pos]
            key = (r.lemma_ru, r.feats, r.dialect)
            if key in cache: pr[i] = cache[key]; continue
            wkey = (r.lemma_ru, r.feats, r.dialect)
            obs = [(f, fm) for f, d, fm in known_map.get(r.lemma_ru, []) if not (f == r.feats and d == r.dialect)]
            seen = collections.defaultdict(list)
            for f, fm in obs:
                c = stress_class(fm)
                if f in cidx and c >= 0: seen[f].append(c)
            ob = tuple(sorted((f, collections.Counter(cs).most_common(1)[0][0]) for f, cs in seen.items()))
            if (r.pos, ob) not in wcache:
                lw = np.zeros(A.shape[0])
                for f, c in ob: lw += LP[:, cidx[f], c]
                w = np.exp(lw - lw.max()); wcache[(r.pos, ob)] = w / w.sum()
            w = wcache[(r.pos, ob)]
            if r.feats in cidx: dist = w @ np.exp(LP[:, cidx[r.feats], :])
            else: dist = np.full(NC, 1.0 / NC)
            cache[key] = dist; pr[i] = dist
        np.save(os.path.join(OUT, f'prior_{seg}_knn{alpha}_lim0.npy'), pr)
        print(f'alpha={alpha} {seg} n={len(rows)} in {time.time()-t:.0f}s', flush=True)
