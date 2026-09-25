"""KAM rows that have a SEV/POM sibling in the holdout: direct KAM prediction vs
(a) rescoring the KAM pool by pair-consistency with the sibling's prediction, (b) adding the rule-mapped sibling prediction.
  python3 src/exp/dialect_consistency/kam_sibling.py runs/exp/dialect_consistency/cache_noloc.pkl
"""
import sys, os, pickle, collections, math
import numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paradigm import stress_class
from validate import load_splits
from pairmodel import PairModel, strip
from kammap import to_kam
cache = pickle.load(open(sys.argv[1], 'rb')); w_loc = 0.5
train_rows, known, val = load_splits()
pm0 = PairModel(ctx=0).fit(train_rows); pm1 = PairModel(ctx=1).fit(train_rows)
for seg in ['wug', 'unseen']:
    if seg not in cache: continue
    c = cache[seg]; df = c['df']; rows = list(df.itertuples())
    pools = []
    for i in range(len(df)):
        k = c['cls'][i]; pool = collections.defaultdict(float)
        for s, q in c['tallies'][i].items():
            if stress_class(s) == k: pool[s] += q
        if c['loc'] is not None:
            for s, q in c['loc'][i].items():
                if stress_class(s) == k: pool[s] += w_loc * q
        if not pool: pool[c['base'][i]] = 1e-3
        pools.append(dict(pool))
    pred = [max(pl.items(), key=lambda x: x[1])[0] for pl in pools]
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    targets = []  # (kam_idx, sib_idx)
    for key, idx in groups.items():
        kam = [i for i in idx if rows[i].dialect == 'KAM']
        sib = [i for i in idx if rows[i].dialect == 'SEV'] or [i for i in idx if rows[i].dialect == 'POM']
        if kam and sib: targets.append((kam[0], sib[0]))
    g = {i: rows[i].form_vvz for i, _ in targets}
    em_direct = np.mean([pred[i] == g[i] for i, _ in targets])
    em_sib = np.mean([pred[j] == rows[j].form_vvz for _, j in targets])
    print(f'{seg}: KAM rows with SEV/POM sibling n={len(targets)}  direct KAM EM {em_direct:.4f}  (sibling row EM {em_sib:.4f})')
    # mapped-sibling exactness (oracle: map the sibling GOLD to KAM and compare with KAM gold)
    for upto in ['yat', 'sk', 'velar']:
        ex = np.mean([to_kam(rows[j].form_vvz, upto=upto) == g[i] for i, j in targets])
        exp_ = np.mean([to_kam(pred[j], upto=upto) == g[i] for i, j in targets])
        print(f'   rule map up to {upto}: map(sibling gold)==KAM gold {ex:.4f} ; map(sibling pred)==KAM gold {exp_:.4f}')
    for name, pm in [('ctx0', pm0), ('ctx1', pm1)]:
        for beta in [0.25, 0.5, 1.0, 2.0]:
            out = {}
            for i, j in targets:
                sp = pred[j]; sd = rows[j].dialect
                best = max(pools[i].items(), key=lambda x: math.log(x[1] + 1e-4) + beta * pm.logp(sp, x[0], sd, 'KAM'))
                out[i] = best[0]
            em = np.mean([out[i] == g[i] for i, _ in targets]); fixed = sum(out[i] == g[i] and pred[i] != g[i] for i, _ in targets); broke = sum(out[i] != g[i] and pred[i] == g[i] for i, _ in targets)
            print(f'   rescore {name} beta={beta}: EM {em:.4f} (fixed {fixed}, broke {broke})')
    for upto in ['sk', 'velar']:
        for alpha in [0.5, 1.0, 2.0]:
            out = {}
            for i, j in targets:
                pl = dict(pools[i]); m = to_kam(pred[j], upto=upto)
                if stress_class(m) == c['cls'][i]:
                    pl[m] = pl.get(m, 0.0) + alpha * pools[j][pred[j]]
                out[i] = max(pl.items(), key=lambda x: x[1])[0]
            em = np.mean([out[i] == g[i] for i, _ in targets]); fixed = sum(out[i] == g[i] and pred[i] != g[i] for i, _ in targets); broke = sum(out[i] != g[i] and pred[i] == g[i] for i, _ in targets)
            print(f'   add mapped sibling ({upto}) alpha={alpha}: EM {em:.4f} (fixed {fixed}, broke {broke})')
