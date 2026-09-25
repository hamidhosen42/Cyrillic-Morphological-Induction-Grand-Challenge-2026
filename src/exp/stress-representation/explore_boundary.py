"""Measure: (1) a lemma-based rule for the stem vowel count, validated against the LCP of the
lemma's own forms; (2) the inventory of stress positions relative to the stem boundary."""
import sys, os, collections, re
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from paradigm import ACC, VOW, stress_idx, n_vow

tr = pd.read_csv(f'{ROOT}/data/train.csv')

def norm(s):
    s = s.replace(ACC, '')
    s = s.replace('о', 'а').replace('ё', 'е').replace('ѣ', 'е').replace('и', 'е').replace('ы', 'е')
    return s

def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def stem_vowels_rule(lemma, pos, dialect):
    l = lemma
    if pos == 'N':
        if l[-1] in 'аяоеё': stem = l[:-1]
        elif l[-1] in 'ьй': stem = l[:-1]
        else: stem = l
    elif pos == 'V':
        if l.endswith('ться') or l.endswith('тись'): stem = l[:-4]
        elif l.endswith('чься'): stem = l[:-4]
        elif l.endswith('ть') or l.endswith('ти') or l.endswith('чь'): stem = l[:-2]
        else: stem = l
    else:
        if l.endswith('ся'): l = l[:-2]
        stem = l[:-2] if l[-2:] in ('ый', 'ий', 'ой') else l
    nv = sum(1 for c in stem if c in VOW)
    if dialect == 'SEV':
        nv -= len(re.findall('оро|ере|оло', stem))
    return nv

# LCP-based stem vowel count per (lemma, dialect): mode over pairs of forms of the min vowel count of common prefix
rows = collections.defaultdict(list)
for l, p, d, fm in zip(tr.lemma_ru, tr.pos, tr.dialect, tr.form_vvz):
    rows[(l, p, d)].append(fm)
agree = collections.Counter(); diffs = collections.Counter(); examples = collections.defaultdict(list)
for (l, p, d), fms in rows.items():
    if len(fms) < 4: continue
    ns = [norm(f) for f in fms]
    # stem string = the prefix shared by the majority: take pairwise lcp lengths, mode of vowel counts of prefix
    cnt = collections.Counter()
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            k = lcp(ns[i], ns[j])
            cnt[sum(1 for c in ns[i][:k] if c in VOW)] += 1
    # the stem's vowel count is the vowel count of the common prefix among "clean" forms: use the max count with >= 30% support
    tot = sum(cnt.values())
    best = max((v for v, c in cnt.items() if c >= 0.3 * tot), default=None)
    if best is None: continue
    r = stem_vowels_rule(l, p, d)
    agree[(p, d, r == best)] += 1
    if r != best:
        diffs[(p, d, r - best)] += 1
        if len(examples[(p, d)]) < 6: examples[(p, d)].append((l, r, best, fms[:5]))
for k in sorted(agree): print('rule==lcp', k, agree[k])
print('diffs (rule - lcp):', sorted(diffs.items(), key=lambda x: -x[1])[:20])
for k, v in examples.items():
    print(k)
    for e in v: print('   ', e)

# position inventory relative to stem boundary (using the rule)
inv = collections.Counter(); inv_cell = collections.defaultdict(collections.Counter)
for l, p, d, f, fm in zip(tr.lemma_ru, tr.pos, tr.dialect, tr.feats, tr.form_vvz):
    si = stress_idx(fm); nv = n_vow(fm); ns = stem_vowels_rule(l, p, d)
    if si < 0: inv['noacc'] += 1; continue
    ne = nv - ns
    if si == 0 and ns == 1: lab = 'I=S'
    elif si == 0: lab = 'I'
    elif si == ns - 1: lab = 'S'
    elif si == ns: lab = 'E'
    elif si < ns - 1: lab = 'M'
    else: lab = 'E2+'
    inv[(p, lab)] += 1
    inv_cell[(p, f, d)][(lab, ne)] += 1
print('inventory', sorted(inv.items()))
# ending vowel count variability per cell
print('ending-vowel-count distribution per cell (top):')
for k in sorted(inv_cell)[:80]:
    c = inv_cell[k]; tot = sum(c.values())
    ne_c = collections.Counter()
    for (lab, ne), v in c.items(): ne_c[ne] += v
    print(k, 'n=%d' % tot, 'ne:', dict(sorted(ne_c.items())), 'labs:', {lab: round(sum(v for (lb, ne), v in c.items() if lb == lab) / tot, 2) for lab in ['I', 'I=S', 'S', 'E', 'M', 'E2+']})
