"""Fix prototype A: ADJ LONG form = SHORT form (same lemma, same dialect) + 'ии' (PL) / 'ыи' (SG).
Applied only as a re-ranker inside the chosen-class candidate pool: if a pool candidate's accent-stripped
form equals the synthesized string, choose it (max prob among such). Otherwise leave the row unchanged.
SHORT source: known form (train rows) in the same dialect, else the pipeline's own prediction for that
SHORT cell in the same dump (same lemma+dialect), else nothing.
  python3 src/exp/residual-errors/fix_adjlong.py runs/exp/residual-errors/preds_base.pkl
"""
import sys, pickle, collections
sys.path.insert(0, 'src')
from paradigm import ACC, stress_class
from validate import load_splits, weighted_score
from rapidfuzz.distance import Levenshtein as Lev

def strip(s): return s.replace(ACC, '')
files = [a for a in sys.argv[1:] if not a.startswith('--')]
use_pred_short = '--no_pred_short' not in sys.argv
safe = '--safe' in sys.argv
def akn(s): return strip(s).replace('о','а').replace('ѣ','е').replace('и','е').replace('ы','е')
recs = []
for f in files: recs += pickle.load(open(f, 'rb'))
train_rows, known, val = load_splits()
kshort = {}
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    kshort[(l, f, d)] = fm
pshort = {}
for r in recs:
    if r['pos'] == 'ADJ' and 'SHORT' in r['feats']:
        pshort[(r['lemma'], r['feats'], r['dialect'])] = r['pred']

stats = collections.defaultdict(collections.Counter)
new = {}
for k, r in enumerate(recs):
    new[k] = r['pred']
    if r['pos'] != 'ADJ' or 'LONG' not in r['feats']: continue
    fs = r['feats'].replace('LONG', 'SHORT'); key = (r['lemma'], fs, r['dialect'])
    src = None
    if key in kshort: src = strip(kshort[key]); how = 'known'
    elif use_pred_short and key in pshort: src = strip(pshort[key]); how = 'pred'
    if src is None: stats[r['seg']]['no_source'] += 1; continue
    target = src + ('ии' if ';PL;' in r['feats'] else 'ыи')
    pool = r['pool'] if r['pool'] else {s: q for s, q in r['tally'].items() if stress_class(s) == r['cls']}
    cands = [(s, q) for s, q in pool.items() if strip(s) == target]
    if not cands: stats[r['seg']]['no_match_' + how] += 1; continue
    best = max(cands, key=lambda x: x[1])[0]
    if best == r['pred']: stats[r['seg']]['same'] += 1; continue
    if safe:
        ops = Lev.opcodes(strip(r['pred']), target)
        ins = [o for o in ops if o.tag != 'equal']
        if not (len(ins) == 1 and ins[0].tag == 'insert' and Lev.distance(akn(r['pred']), akn(target)) == 1):
            stats[r['seg']]['skipped_unsafe'] += 1; continue
    new[k] = best
    was = r['pred'] == r['gold']; now = best == r['gold']
    stats[r['seg']]['changed'] += 1
    if now and not was: stats[r['seg']]['fixed_' + how] += 1
    elif was and not now: stats[r['seg']]['broke_' + how] += 1
    else: stats[r['seg']]['still_wrong'] += 1
for seg in ['transfer', 'complete', 'wug', 'unseen']:
    if stats[seg]: print(seg, dict(stats[seg]))
segs = sorted({r['seg'] for r in recs})
pb = {s: [r['pred'] for r in recs if r['seg'] == s] for s in segs}
pn = {s: [new[k] for k, r in enumerate(recs) if r['seg'] == s] for s in segs}
g = {s: [r['gold'] for r in recs if r['seg'] == s] for s in segs}
print('before', weighted_score(pb, g)); print('after ', weighted_score(pn, g))
pickle.dump(new, open('runs/exp/residual-errors/fixA_new.pkl', 'wb'))
