"""Joint decoding across dialect rows of the same (lemma, feats) using the cached pools.
  python3 src/exp/dialect_consistency/joint_dialect.py --cache runs/exp/dialect_consistency/cache.pkl --segs wug,unseen --beta 0.5,1,2
"""
import argparse, os, sys, pickle, collections, math, itertools
import numpy as np, pandas as pd
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paradigm import stress_class
from validate import load_splits, weighted_score, row_scores
from pairmodel import PairModel, strip

p = argparse.ArgumentParser()
p.add_argument('--cache', required=True)
p.add_argument('--segs', default='wug,unseen')
p.add_argument('--beta', default='0.5,1,2')
p.add_argument('--w_loc', type=float, default=0.5)
p.add_argument('--topm', type=int, default=8)
p.add_argument('--ctx', type=int, default=0)
p.add_argument('--floor', type=float, default=0.5)
p.add_argument('--alpha', default='0')      # sibling-candidate augmentation weight (identity mapping)
p.add_argument('--eps', type=float, default=1e-4)
p.add_argument('--dump', default=None)
args = p.parse_args()

cache = pickle.load(open(args.cache, 'rb'))
train_rows, known, val = load_splits()
pm = PairModel(ctx=args.ctx, floor=args.floor).fit(train_rows)
betas = [float(b) for b in args.beta.split(',')]
alphas = [float(a) for a in args.alpha.split(',')]


def pools_for(seg):
    c = cache[seg]
    pools = []
    for i in range(len(c['df'])):
        k = c['cls'][i]
        pool = collections.defaultdict(float)
        for s, q in c['tallies'][i].items():
            if stress_class(s) == k: pool[s] += q
        if c['loc'] is not None:
            for s, q in c['loc'][i].items():
                if stress_class(s) == k: pool[s] += args.w_loc * q
        if not pool:
            pool[c['base'][i]] = 1e-3
        pools.append(dict(pool))
    return pools


def joint(seg, beta, alpha):
    c = cache[seg]; df = c['df']; pools = pools_for(seg)
    rows = list(df.itertuples())
    merged = [max(pl.items(), key=lambda x: x[1])[0] for pl in pools]
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    out = list(merged); nchanged = 0
    for key, idx in groups.items():
        if len(idx) < 2:
            continue
        ds = [rows[i].dialect for i in idx]
        # candidate augmentation: sibling candidates (identity mapping) scaled by P(identical | dialect pair)
        pls = []
        for a, i in enumerate(idx):
            pl = dict(pools[i])
            if alpha > 0:
                for b, j in enumerate(idx):
                    if j == i: continue
                    pid = pm.ident[pm.key(ds[a], ds[b])]
                    for s, q in pools[j].items():
                        pl[s] = pl.get(s, 0.0) + alpha * pid * q
            pls.append(pl)
        tops = [sorted(pl.items(), key=lambda x: -x[1])[:args.topm] for pl in pls]
        best, bs = None, -1e18
        for tup in itertools.product(*tops):
            sc = sum(math.log(q + args.eps) for _, q in tup)
            if beta:
                for a in range(len(tup)):
                    for b in range(a + 1, len(tup)):
                        sc += beta * pm.logp(tup[a][0], tup[b][0], ds[a], ds[b])
            if sc > bs: best, bs = tup, sc
        for a, i in enumerate(idx):
            if best[a][0] != out[i]: nchanged += 1
            out[i] = best[a][0]
    return merged, out, nchanged


segs = args.segs.split(',')
golds = {s: list(cache[s]['df'].form_vvz) for s in segs}
res = {}
for seg in segs:
    base = cache[seg]['base']
    em_b = np.mean([a == g for a, g in zip(base, golds[seg])])
    print(f'{seg:8s} n={len(golds[seg])} base EM {em_b:.4f}')
    for alpha in alphas:
        for beta in betas:
            merged, out, nch = joint(seg, beta, alpha)
            em_m = np.mean([a == g for a, g in zip(merged, golds[seg])])
            em_j = np.mean([a == g for a, g in zip(out, golds[seg])])
            res[(seg, alpha, beta)] = (merged, out)
            print(f'   alpha={alpha} beta={beta}: merged EM {em_m:.4f}  joint EM {em_j:.4f}  delta {em_j-em_m:+.4f} ({int(round((em_j-em_m)*len(out)))} rows) changed {nch}', flush=True)
if args.dump:
    pickle.dump(res, open(args.dump, 'wb'))
