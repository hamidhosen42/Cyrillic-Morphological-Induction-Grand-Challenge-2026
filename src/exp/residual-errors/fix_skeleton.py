"""Fix prototype B: lemma consonant-skeleton consistency re-ranker (targets stem-copy errors in long words).
cons(x): remove accent, vowels, ь/ъ/-, collapse palatalization (ц,ч->к; з,ж->г; с,ш->х).
A candidate is 'consistent' if cons(candidate) starts with cons(lemma_stem).
Only rows whose current pred is inconsistent are changed, to the max-prob consistent candidate in the chosen-class pool.
  python3 src/exp/residual-errors/fix_skeleton.py runs/exp/residual-errors/preds_base.pkl
"""
import sys, pickle, collections, re
sys.path.insert(0, 'src')
from paradigm import ACC, stress_class
from validate import weighted_score

VOW = 'аеиоуыэюяёѣѫ'
def strip(s): return s.replace(ACC, '')
def cons(s):
    s = strip(s).replace('-', '')
    s = re.sub('[' + VOW + 'ьъй]', '', s)
    return s.translate(str.maketrans('цчзжсш', 'ккггхх'))
def lemma_stem(lemma, pos):
    l = lemma
    if pos == 'ADJ':
        for e in ('ый', 'ий', 'ой'):
            if l.endswith(e): return l[:-len(e)]
    if pos == 'V':
        for e in ('ться', 'ти', 'ть', 'чь'):
            if l.endswith(e): return l[:-len(e)]
    if pos == 'N':
        for e in ('ия', 'ие', 'а', 'я', 'о', 'е', 'ь', 'й'):
            if l.endswith(e): return l[:-len(e)]
    return l
files = [a for a in sys.argv[1:] if not a.startswith('--')]
only_pos = None
for a in sys.argv[1:]:
    if a.startswith('--pos='): only_pos = a[6:].split(',')
recs = []
for f in files: recs += pickle.load(open(f, 'rb'))
stats = collections.defaultdict(collections.Counter); new = {}
for k, r in enumerate(recs):
    new[k] = r['pred']
    if only_pos and r['pos'] not in only_pos: continue
    ls = cons(lemma_stem(r['lemma'], r['pos']))
    if cons(r['pred']).startswith(ls): stats[r['seg']]['pred_ok'] += 1; continue
    stats[r['seg']]['pred_inconsistent'] += 1
    stats[r['seg']]['gold_inconsistent_too'] += int(not cons(r['gold']).startswith(ls))
    pool = r['pool'] if r['pool'] else {s: q for s, q in r['tally'].items() if stress_class(s) == r['cls']}
    cands = [(s, q) for s, q in pool.items() if cons(s).startswith(ls)]
    if not cands: stats[r['seg']]['no_consistent_cand'] += 1; continue
    best = max(cands, key=lambda x: x[1])[0]
    new[k] = best
    was = r['pred'] == r['gold']; now = best == r['gold']
    stats[r['seg']]['changed'] += 1
    if now and not was: stats[r['seg']]['fixed'] += 1
    elif was and not now: stats[r['seg']]['broke'] += 1
    else: stats[r['seg']]['still_wrong'] += 1
for seg in ['transfer', 'complete', 'wug', 'unseen']:
    if stats[seg]: print(seg, dict(stats[seg]))
segs = sorted({r['seg'] for r in recs})
pb = {s: [r['pred'] for r in recs if r['seg'] == s] for s in segs}
pn = {s: [new[k] for k, r in enumerate(recs) if r['seg'] == s] for s in segs}
g = {s: [r['gold'] for r in recs if r['seg'] == s] for s in segs}
print('before', weighted_score(pb, g)); print('after ', weighted_score(pn, g))
pickle.dump(new, open('runs/exp/residual-errors/fixB_new.pkl', 'wb'))
