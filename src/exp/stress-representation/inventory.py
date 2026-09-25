"""Stress-position inventory relative to the lemma-rule stem boundary, and per-cell ending vowel counts."""
import sys, os, collections, re
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from paradigm import ACC, VOW, stress_idx, n_vow
sys.path.insert(0, os.path.dirname(__file__))
from stemrule import stem_vowels_rule

tr = pd.read_csv(f'{ROOT}/data/train.csv')
inv = collections.Counter(); inv_cell = collections.defaultdict(collections.Counter); nv_cell = collections.defaultdict(collections.Counter)
for l, p, d, f, fm in zip(tr.lemma_ru, tr.pos, tr.dialect, tr.feats, tr.form_vvz):
    si = stress_idx(fm); nv = n_vow(fm); ns = stem_vowels_rule(l, p, d)
    if si < 0: inv[p + ':noacc'] += 1; continue
    ne = nv - ns
    if si == 0 and ns == 1: lab = 'I=S'
    elif si == 0: lab = 'I'
    elif si == ns - 1: lab = 'S'
    elif si == ns: lab = 'E'
    elif si < ns - 1: lab = 'M'
    else: lab = 'E2+'
    inv[p + ':' + lab] += 1
    inv_cell[(p, f, d)][lab] += 1
    nv_cell[(p, f, d)][ne] += 1
tot = collections.Counter()
for k, v in inv.items(): tot[k.split(':')[0]] += v
for k in sorted(inv): print('inventory', k, inv[k], round(inv[k] / tot[k.split(':')[0]], 4))
print('per cell: ending vowel counts and label mix')
for k in sorted(inv_cell):
    c = inv_cell[k]; n = sum(c.values())
    print(k, 'n=%d' % n, 'ne:', dict(sorted(nv_cell[k].items())), 'labs:', {lab: round(c[lab] / n, 2) for lab in ['I', 'I=S', 'S', 'E', 'M', 'E2+'] if c[lab]})
