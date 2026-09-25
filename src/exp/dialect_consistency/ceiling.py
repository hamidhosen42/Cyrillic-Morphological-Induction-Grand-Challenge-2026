"""Ceiling analysis: within multi-dialect groups, how often is a row wrong while a sibling row is right,
and is the row's gold segmentally derivable from the sibling's gold (identity or learned op set)?"""
import sys, os, pickle, collections, math
import numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paradigm import stress_class, stress_idx
from pairmodel import strip, ops
cache = pickle.load(open(sys.argv[1], 'rb'))
for seg in ['wug', 'unseen', 'transfer', 'complete']:
    if seg not in cache: continue
    c = cache[seg]; df = c['df']; rows = list(df.itertuples())
    pools = []
    for i in range(len(df)):
        k = c['cls'][i]; pool = collections.defaultdict(float)
        for s, q in c['tallies'][i].items():
            if stress_class(s) == k: pool[s] += q
        if c['loc'] is not None:
            for s, q in c['loc'][i].items():
                if stress_class(s) == k: pool[s] += 0.5 * q
        pools.append(dict(pool))
    pred = [max(pl.items(), key=lambda x: x[1])[0] if pl else c['base'][i] for i, pl in enumerate(pools)]
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    st = collections.Counter(); n_multi = 0
    err_kind = collections.Counter()
    for key, idx in groups.items():
        if len(idx) < 2: continue
        n_multi += len(idx)
        ok = [pred[i] == rows[i].form_vvz for i in idx]
        for a, i in enumerate(idx):
            sib_ok = any(ok[b] for b in range(len(idx)) if b != a)
            st[(ok[a], sib_ok)] += 1
            if not ok[a] and sib_ok:
                g = rows[i].form_vvz; p = pred[i]
                # classify the error of row i
                if strip(g) == strip(p): kind = 'stress-only'
                elif stress_class(g) != stress_class(p): kind = 'class+seg'
                else: kind = 'seg-only'
                # is gold identical (stripped) to a correct sibling's gold?
                sibs = [rows[idx[b]].form_vvz for b in range(len(idx)) if b != a and ok[b]]
                ident = any(strip(s) == strip(g) for s in sibs)
                in_pool = g in pools[i]
                err_kind[(kind, 'sib_ident' if ident else 'sib_diff', 'gold_in_pool' if in_pool else 'gold_not_in_pool')] += 1
    print(f'{seg}: rows in multi-dialect groups {n_multi}/{len(rows)}; (row_ok, sibling_ok) counts: {dict(st)}')
    print('   row wrong & sibling right, by kind:', sorted(err_kind.items(), key=lambda x: -x[1]))
