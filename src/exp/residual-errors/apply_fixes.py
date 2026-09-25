"""Apply the three prototype post-fixes sequentially (B skeleton -> A adj-long safe -> C geminates) to a set of
prediction dumps and report per-segment EM / weighted score before and after, plus fixed/broken counts.
  python3 src/exp/residual-errors/apply_fixes.py runs/exp/residual-errors/preds_base.pkl
  python3 src/exp/residual-errors/apply_fixes.py runs/exp/residual-errors/preds_{wug,unseen,complete,transfer_*}.pkl
"""
import sys, pickle, collections, re, importlib.util, os
sys.path.insert(0, 'src')
from paradigm import ACC, stress_class
from validate import load_splits, weighted_score
from rapidfuzz.distance import Levenshtein as Lev
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('fixgem', os.path.join(HERE, 'fix_gemination.py')); fixgem = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixgem)

VOW = 'аеиоуыэюяёѣѫ'
def strip(s): return s.replace(ACC, '')
def akn(s): return strip(s).replace('о', 'а').replace('ѣ', 'е').replace('и', 'е').replace('ы', 'е')
def cons(s):
    s = strip(s).replace('-', ''); s = re.sub('[' + VOW + 'ьъй]', '', s)
    return s.translate(str.maketrans('цчзжсш', 'ккггхх'))
def lemma_stem(lemma, pos):
    l = lemma
    ends = {'ADJ': ('ый', 'ий', 'ой'), 'V': ('ться', 'ти', 'ть', 'чь'), 'N': ('ия', 'ие', 'а', 'я', 'о', 'е', 'ь', 'й')}[pos]
    for e in ends:
        if l.endswith(e): return l[:-len(e)]
    return l
def pool_of(r):
    return r['pool'] if r['pool'] else {s: q for s, q in r['tally'].items() if stress_class(s) == r['cls']}

def fix_B(recs, cur):
    out = dict(cur)
    for k, r in enumerate(recs):
        ls = cons(lemma_stem(r['lemma'], r['pos']))
        if cons(cur[k]).startswith(ls): continue
        cands = [(s, q) for s, q in pool_of(r).items() if cons(s).startswith(ls)]
        if cands: out[k] = max(cands, key=lambda x: x[1])[0]
    return out

def fix_A(recs, cur, kshort):
    pshort = {(r['lemma'], r['feats'], r['dialect']): cur[k] for k, r in enumerate(recs) if r['pos'] == 'ADJ' and 'SHORT' in r['feats']}
    out = dict(cur)
    for k, r in enumerate(recs):
        if r['pos'] != 'ADJ' or 'LONG' not in r['feats']: continue
        key = (r['lemma'], r['feats'].replace('LONG', 'SHORT'), r['dialect'])
        src = strip(kshort[key]) if key in kshort else (strip(pshort[key]) if key in pshort else None)
        if src is None: continue
        target = src + ('ии' if ';PL;' in r['feats'] else 'ыи')
        cands = [(s, q) for s, q in pool_of(r).items() if strip(s) == target]
        if not cands: continue
        best = max(cands, key=lambda x: x[1])[0]
        if best == cur[k]: continue
        ins = [o for o in Lev.opcodes(strip(cur[k]), target) if o.tag != 'equal']
        if len(ins) == 1 and ins[0].tag == 'insert' and Lev.distance(akn(cur[k]), akn(target)) == 1:
            out[k] = best
    return out

def fix_C(recs, cur):
    return {k: fixgem.fix(r['lemma'], cur[k]) for k, r in enumerate(recs)}

def report(name, recs, before, after):
    st = collections.defaultdict(collections.Counter)
    for k, r in enumerate(recs):
        if before[k] == after[k]: continue
        was = before[k] == r['gold']; now = after[k] == r['gold']
        st[r['seg']]['changed'] += 1
        st[r['seg']]['fixed' if (now and not was) else 'broke' if (was and not now) else 'still_wrong'] += 1
    print(f'-- {name}: ' + '; '.join(f"{s}: {dict(st[s])}" for s in ['transfer', 'complete', 'wug', 'unseen'] if st[s]))

if __name__ == '__main__':
    files = [a for a in sys.argv[1:] if not a.startswith('--')]
    recs = []
    for f in files: recs += pickle.load(open(f, 'rb'))
    _, known, _ = load_splits()
    kshort = {(l, f, d): fm for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz)}
    p0 = {k: r['pred'] for k, r in enumerate(recs)}
    p1 = fix_B(recs, p0); report('B skeleton', recs, p0, p1)
    p2 = fix_A(recs, p1, kshort); report('A adj-long', recs, p1, p2)
    p3 = fix_C(recs, p2); report('C geminate', recs, p2, p3)
    report('ALL', recs, p0, p3)
    segs = [s for s in ['transfer', 'complete', 'wug', 'unseen'] if any(r['seg'] == s for r in recs)]
    g = {s: [r['gold'] for r in recs if r['seg'] == s] for s in segs}
    for name, p in [('before', p0), ('after ', p3)]:
        print(name, weighted_score({s: [p[k] for k, r in enumerate(recs) if r['seg'] == s] for s in segs}, g))
    print('n per seg', {s: len(g[s]) for s in segs})
    pickle.dump(p3, open('runs/exp/residual-errors/fixALL_new.pkl', 'wb'))
