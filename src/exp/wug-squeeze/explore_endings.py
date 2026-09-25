"""Segment forms into stem+ending via per-(lemma,dialect) longest common prefix; build ending inventory per cell."""
import pandas as pd, numpy as np, collections, sys, os, pickle
sys.path.insert(0, 'src')
from paradigm import stress_class, stress_idx, n_vow, ACC, VOW
tr = pd.read_csv('data/train.csv'); wug = pd.read_csv('data/wug_seeds.csv')
tr['plain'] = tr.form_vvz.str.replace(ACC, '', regex=False)
def lcp(strs):
    s = min(strs, key=len); i = 0
    while i < len(s) and all(t[i] == s[i] for t in strs): i += 1
    return i
inv = collections.defaultdict(collections.Counter)  # (pos,feats,dialect) -> Counter(ending)
for (lem, dia), d in tr.groupby(['lemma_ru', 'dialect']):
    if len(d) < 3: continue
    L = lcp(list(d.plain))
    if L < 2: continue
    for f, p, pos in zip(d.feats, d.plain, d.pos):
        inv[(pos, f, dia)][p[L:]] += 1
for key in sorted(inv):
    c = inv[key]; tot = sum(c.values())
    print(key, tot, [(e, round(n / tot, 3)) for e, n in c.most_common(8)])
pickle.dump(inv, open('runs/exp/wug-squeeze/ending_inv.pkl', 'wb'))
# seed endings
w = wug.copy(); w['plain'] = w.form_vvz.str.replace(ACC, '', regex=False)
for pos in ['N', 'V']:
    print(pos, 'wug seed last2', collections.Counter(w[w.pos == pos].plain.str[-2:]).most_common(15))
    print(pos, 'wug seed nv-lemnv', collections.Counter(w[w.pos == pos].apply(lambda r: n_vow(r.form_vvz) - n_vow(r.lemma_ru), axis=1)).most_common())
