"""Deterministic dialect-orthography constraints (measured in train: KAM has no unstressed ѣ and no 'сьц';
SEV/POM have no 'ськ'). Test (a) post-fix KAM unstressed ѣ->и, (b) pool filter dropping violating candidates.
  python3 src/exp/dialect_consistency/constraints.py runs/exp/dialect_consistency/cache_noloc.pkl [w_loc]
"""
import sys, os, pickle, collections
import numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, SRC)
from paradigm import stress_class
from validate import weighted_score
ACC = '́'
cache = pickle.load(open(sys.argv[1], 'rb')); w_loc = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


def unstressed_yat(s):
    return any(ch == 'ѣ' and not (i + 1 < len(s) and s[i + 1] == ACC) for i, ch in enumerate(s))


def violates(s, d):
    if d == 'KAM':
        return unstressed_yat(s) or 'сьц' in s
    return 'ськ' in s


def fix_kam(s):
    out = list(s)
    for i, ch in enumerate(out):
        if ch == 'ѣ' and not (i + 1 < len(out) and out[i + 1] == ACC): out[i] = 'и'
    return ''.join(out)


preds = {'merged': {}, 'postfix': {}, 'filter': {}, 'filter+postfix': {}}; golds = {}
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
                if stress_class(s) == k: pool[s] += w_loc * q
        if not pool: pool[c['base'][i]] = 1e-3
        pools.append(dict(pool))
    merged = [max(pl.items(), key=lambda x: x[1])[0] for pl in pools]
    postfix = [fix_kam(p) if r.dialect == 'KAM' else p for p, r in zip(merged, rows)]
    filt = []
    for i, (pl, r) in enumerate(zip(pools, rows)):
        ok = {s: q for s, q in pl.items() if not violates(s, r.dialect)}
        filt.append(max(ok.items(), key=lambda x: x[1])[0] if ok else merged[i])
    fp = [fix_kam(p) if r.dialect == 'KAM' else p for p, r in zip(filt, rows)]
    golds[seg] = list(df.form_vvz)
    for name, pr in [('merged', merged), ('postfix', postfix), ('filter', filt), ('filter+postfix', fp)]:
        preds[name][seg] = pr
    g = golds[seg]
    nviol = sum(violates(p, r.dialect) for p, r in zip(merged, rows))
    print(f'{seg:9s} n={len(g)} merged EM {np.mean([a==b for a,b in zip(merged,g)]):.4f} | violating preds {nviol} | postfix EM {np.mean([a==b for a,b in zip(postfix,g)]):.4f} (+{sum(a==b for a,b in zip(postfix,g))-sum(a==b for a,b in zip(merged,g))} rows) | filter EM {np.mean([a==b for a,b in zip(filt,g)]):.4f} (+{sum(a==b for a,b in zip(filt,g))-sum(a==b for a,b in zip(merged,g))} rows) | filter+postfix (+{sum(a==b for a,b in zip(fp,g))-sum(a==b for a,b in zip(merged,g))} rows)')
if len(golds) == 4:
    for name in preds:
        print(name, weighted_score(preds[name], golds))
