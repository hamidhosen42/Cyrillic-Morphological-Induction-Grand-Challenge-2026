"""Recover the shuffle seed of dev.csv: candidate permutation vs canonical (rank, cell, dialect) order of dev rows."""
import pandas as pd, numpy as np, random, time, sys, argparse, pickle
ap = argparse.ArgumentParser(); ap.add_argument('--fam', default='np'); ap.add_argument('--s0', type=int, default=0); ap.add_argument('--s1', type=int, default=20000)
ap.add_argument('--Ns', default='20000,320000,380000'); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--out', default='runs/exp/rng_paradigm/devperm.pkl')
args = ap.parse_args()
tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv')
L = pd.read_pickle('runs/exp/rng_paradigm/lemmas_lab.pkl'); rank = dict(zip(L.lemma_ru, L['rank']))
# canonical cell order from train within-lemma positions
tr['pos_in_lemma'] = tr.groupby('lemma_ru').cumcount()
cellpos = tr.groupby('feats').pos_in_lemma.mean().sort_values(); cellidx = {c: i for i, c in enumerate(cellpos.index)}
DI = {'SEV': 0, 'POM': 1, 'KAM': 2}
def key(df): return df.lemma_ru.map(rank).values * 1000 + df.feats.map(cellidx).values * 10 + df.dialect.map(DI).values
kt = key(tr); print('train sorted by canonical key:', bool((np.diff(kt) > 0).all()), 'violations', (np.diff(kt) <= 0).sum())
kd = key(dv); canon = np.argsort(np.argsort(kd)).astype(float)   # canonical rank of dev row i among dev rows
n = args.n; c = canon[:n]; c = (c - c.mean()) / c.std()
def rho(p):
    r = np.argsort(np.argsort(p[:n])).astype(float); r = (r - r.mean()) / r.std(); return float((r * c).mean())
res = []; t0 = time.time(); best = (0, None)
Ns = [int(x) for x in args.Ns.split(',')]
for s in range(args.s0, args.s1):
    for N in Ns:
        if args.fam == 'np': p = np.random.RandomState(s).permutation(N)
        elif args.fam == 'gen': p = np.random.default_rng(s).permutation(N)
        elif args.fam == 'pyshuf': x = list(range(N)); random.Random(s).shuffle(x); p = np.array(x)
        elif args.fam == 'pysample': p = np.array(random.Random(s).sample(range(N), min(N, 20000)))
        r = rho(p)
        if abs(r) > abs(best[0]): best = (r, (s, N))
        if abs(r) > 0.2: res.append((s, N, r)); print('HIT', s, N, r, flush=True)
    if s % 2000 == 0 and s > args.s0: print(args.fam, s, f't={time.time()-t0:.0f}s best', best, flush=True)
print(args.fam, 'done; best |rho|', best, 'null sd', 1 / np.sqrt(n), 'hits', res, 'time', round(time.time() - t0))
pickle.dump(dict(best=best, hits=res, args=vars(args)), open(args.out, 'wb'))
