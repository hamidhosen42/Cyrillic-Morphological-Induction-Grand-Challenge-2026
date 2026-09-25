"""Categorize every holdout error of the current best pipeline.
  python3 src/exp/residual-errors/categorize.py runs/exp/residual-errors/preds_*.pkl
"""
import sys, pickle, collections, glob, re
import numpy as np
sys.path.insert(0, 'src')
from paradigm import stress_class, ACC
from rapidfuzz.distance import Levenshtein as Lev
from validate import W, TEST_N

SHARE = {'transfer': .131, 'complete': .218, 'wug': .398, 'unseen': .254}

def strip(s): return s.replace(ACC, '')

def norm(s):
    """crude normalisation to compare stem identity: remove accent, akanye, yat, KAM-и, pleophony"""
    s = strip(s).replace('ѣ', 'е').replace('ѫ', 'у').replace('ъ', '').replace('ь', '')
    s = s.replace('о', 'а').replace('и', 'е').replace('ы', 'е').replace('ё', 'е')
    s = s.replace('ра', 'ара').replace('ла', 'ала').replace('ре', 'ере')
    s = s.replace('ц', 'к').replace('ч', 'к').replace('ж', 'г').replace('з', 'г').replace('ш', 'х').replace('с', 'х')
    return s

def is_noise(r):
    """gold looks like a random-word substitution: stem shares almost nothing with lemma / pred."""
    g, p, l = strip(r['gold']), strip(r['pred']), r['lemma']
    ng, np_, nl = norm(g), norm(p), norm(l)
    # compare first 4 chars of normalized stem with lemma
    k = min(4, len(nl), len(ng))
    if ng[:k] == nl[:k] or ng[:k] == np_[:k]: return False
    d = Lev.normalized_distance(ng, np_)
    return d > 0.4

def diff_ops(a, b):
    """return list of (op, a_sub, b_sub) from Levenshtein opcodes between a and b (accent-stripped)."""
    ops = Lev.opcodes(a, b)
    out = []
    for o in ops:
        if o.tag == 'equal': continue
        out.append((o.tag, a[o.src_start:o.src_end], b[o.dest_start:o.dest_end]))
    return out

PAL = {('к', 'ц'), ('к', 'ч'), ('г', 'з'), ('г', 'ж'), ('х', 'с'), ('х', 'ш'), ('ск', 'сьц'), ('ц', 'ч'), ('ск', 'ськ'), ('сьц', 'ськ'), ('ц', 'к'), ('ч', 'к'), ('з', 'г'), ('ж', 'г'), ('с', 'х'), ('ш', 'х'), ('ськ', 'сьц'), ('сьц', 'ск'), ('ськ', 'ск'), ('ч', 'ц')}

def seg_bucket(r):
    """right class, wrong segmental form -> bucket."""
    g, p = strip(r['gold']), strip(r['pred'])
    if is_noise(r): return 'gold_noise'
    if '-' in r['lemma'] or len(r['lemma']) >= 14: return 'hyphen_long'
    ops = diff_ops(p, g)
    subs = [(a, b) for t, a, b in ops]
    # typo in gold: single char difference, gold form not consistent with its own paradigm mates? (approx: 1 edit and pred == accent-stripped form seen for lemma in other dialect)
    joined_a = ''.join(a for a, b in subs); joined_b = ''.join(b for a, b in subs)
    # gemination: differ only by doubled consonant
    if all((t in ('insert', 'delete')) and len(a + b) == 1 and (a + b) not in 'аеиоуыэюяёѣѫ' for t, a, b in ops):
        # check doubled: the inserted/deleted char neighbours itself
        ok = True
        for t, a, b in ops:
            c = a + b
            src = g if t == 'insert' else p
            if c + c not in src: ok = False
        if ok: return 'gemination'
    if any((a, b) in PAL or (b, a) in PAL for a, b in subs): return 'palatalization'
    # pleophony reversal in SEV
    if r['dialect'] == 'SEV' and any(x in p or x in g for x in ('оро', 'ере', 'оло', 'ра', 'рѣ', 'ла')) and any(
            (a, b) in {('о', ''), ('', 'о'), ('е', ''), ('', 'е'), ('оро', 'ра'), ('ра', 'оро'), ('ере', 'рѣ'), ('рѣ', 'ере'), ('оло', 'ла'), ('ла', 'оло')} for a, b in subs):
        return 'pleophony'
    if r['dialect'] == 'KAM' and all(set(a + b) <= set('еи') for a, b in subs): return 'kam_vowel'
    if all(set(a + b) <= set('ае') for a, b in subs) and all(t == 'replace' for t, a, b in ops): return 'akanye_ae'
    if all(set(a + b) <= set('еѣ') for a, b in subs): return 'yat'
    # ending vs stem
    L = min(len(g), len(p)); i = 0
    while i < L and g[i] == p[i]: i += 1
    tail = len(g) - i
    if r['pos'] == 'V' and tail <= 4 and i >= len(g) - 5: return 'verb_ending'
    if r['pos'] != 'V' and tail <= 3 and i >= len(g) - 4: return 'ending'
    if sum(len(a) + len(b) for a, b in subs) <= 2: return 'stem_1edit'
    return 'other'

def categorize(r):
    if r['pred'] == r['gold']: return 'correct'
    if r['gold_cls'] < 0: return 'gold_noaccent'
    if is_noise(r): return 'gold_noise'
    if r['cls'] != r['gold_cls']:
        return 'class_' + ('goldInTally' if r['gold_in_tally'] else ('goldInLoc' if r['gold_in_loc'] else 'goldNotInCands'))
    return 'seg_' + seg_bucket(r)

if __name__ == '__main__':
    files = sys.argv[1:]
    recs = []
    for f in files: recs += pickle.load(open(f, 'rb'))
    segs = ['transfer', 'complete', 'wug', 'unseen']
    cat = collections.defaultdict(collections.Counter); n = collections.Counter()
    for r in recs:
        r['cat'] = categorize(r); cat[r['seg']][r['cat']] += 1; n[r['seg']] += 1
    print(f"{'bucket':26s}" + ''.join(f"{s:>22s}" for s in segs))
    allc = sorted({c for s in segs for c in cat[s]}, key=lambda c: -sum(cat[s][c] for s in segs))
    for c in allc:
        row = f"{c:26s}"
        for s in segs:
            k = cat[s][c]; frac = k / max(1, n[s]); lb = 0.9 * SHARE[s] * frac
            row += f"{k:6d} {frac*100:5.1f}% {lb:.4f}"
        print(row)
    print('n', dict(n))
    print('EM', {s: round(cat[s]['correct'] / n[s], 4) for s in segs if n[s]})
    pickle.dump(recs, open('runs/exp/residual-errors/preds_cat.pkl', 'wb'))
