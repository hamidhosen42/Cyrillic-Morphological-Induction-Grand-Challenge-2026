import sys, os, pickle, collections, math, itertools
import numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paradigm import stress_class
from validate import load_splits
from pairmodel import PairModel, strip
cache = pickle.load(open(sys.argv[1], 'rb')); seg = sys.argv[2]; beta = float(sys.argv[3]); w_loc = 0.5
train_rows, known, val = load_splits()
pm = PairModel().fit(train_rows)
c = cache[seg]; df = c['df']; rows = list(df.itertuples())
pools = []
for i in range(len(df)):
    k = c['cls'][i]; pool = collections.defaultdict(float)
    for s, q in c['tallies'][i].items():
        if stress_class(s) == k: pool[s] += q
    if c['loc'] is not None:
        for s, q in c['loc'][i].items():
            if stress_class(s) == k: pool[s] += w_loc * q
    pools.append(dict(pool))
groups = collections.defaultdict(list)
for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
shown = 0
stats = collections.Counter()
for key, idx in groups.items():
    if len(idx) < 2: continue
    ds = [rows[i].dialect for i in idx]
    tops = [sorted(pools[i].items(), key=lambda x: -x[1])[:8] for i in idx]
    merged = [t[0][0] if t else None for t in tops]
    best, bs = None, -1e18
    for tup in itertools.product(*tops):
        sc = sum(math.log(q + 1e-4) for _, q in tup)
        for a in range(len(tup)):
            for b in range(a + 1, len(tup)):
                sc += beta * pm.logp(tup[a][0], tup[b][0], ds[a], ds[b])
        if sc > bs: best, bs = tup, sc
    golds = [rows[i].form_vvz for i in idx]
    ch = [a for a in range(len(idx)) if best[a][0] != merged[a]]
    if not ch: continue
    for a in ch:
        stats[('merged_ok', merged[a] == golds[a], 'joint_ok', best[a][0] == golds[a])] += 1
    if shown < 12:
        shown += 1
        print('==', key, ds)
        for a, i in enumerate(idx):
            print(f'  {ds[a]} gold={golds[a]} merged={merged[a]} joint={best[a][0]}  pool={[(s, round(q,3)) for s,q in tops[a][:4]]}')
        for a in range(len(idx)):
            for b in range(a+1, len(idx)):
                print(f'   pair logp gold: {pm.logp(golds[a], golds[b], ds[a], ds[b]):.2f}  merged: {pm.logp(merged[a], merged[b], ds[a], ds[b]):.2f}  joint: {pm.logp(best[a][0], best[b][0], ds[a], ds[b]):.2f}')
print(stats)
