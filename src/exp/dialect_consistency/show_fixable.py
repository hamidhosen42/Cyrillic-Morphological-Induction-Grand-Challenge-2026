import sys, os, pickle, collections
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paradigm import stress_class
from pairmodel import strip
cache = pickle.load(open(sys.argv[1], 'rb')); seg = sys.argv[2]; nshow = int(sys.argv[3])
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
shown = 0
gold_class_same = collections.Counter()
for key, idx in groups.items():
    if len(idx) < 2: continue
    ok = [pred[i] == rows[i].form_vvz for i in idx]
    if all(ok) or not any(ok): continue
    gc = [stress_class(rows[i].form_vvz) for i in idx]
    gold_class_same[len(set(gc)) == 1] += 1
    if shown < nshow:
        shown += 1
        print('==', key, 'cls chosen', c['cls'][idx[0]], 'gold classes', gc)
        for a, i in enumerate(idx):
            print(f'   {rows[i].dialect} {"OK " if ok[a] else "ERR"} gold={rows[i].form_vvz:22s} pred={pred[i]:22s} pool={[(s, round(q,3)) for s,q in sorted(pools[i].items(), key=lambda x:-x[1])[:3]]}')
print('groups with mixed correctness: gold class same across dialects?', dict(gold_class_same))
