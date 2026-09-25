"""Fix prototype C: restore lemma geminates. If the Russian lemma contains a doubled consonant 'cc' and the
prediction contains the aligned single 'c' (with matching left context of >=2 consonant-skeleton letters),
double it. Pure string post-fix; independent of candidate pools. Also reports train-data safety
(how often gold keeps the lemma's geminate).
  python3 src/exp/residual-errors/fix_gemination.py runs/exp/residual-errors/preds_base.pkl
"""
import sys, pickle, collections, re
import pandas as pd
sys.path.insert(0, 'src')
from paradigm import ACC
from validate import weighted_score
CONS = 'бвгджзклмнпрстфхцчшщ'
PAL = {'к': 'цч', 'г': 'зж', 'х': 'сш', 'ц': 'ч'}
def strip(s): return s.replace(ACC, '')
def geminates(lemma):
    return [(m.start(), m.group(1)) for m in re.finditer(r'([' + CONS + r'])\1', lemma)]
def fix(lemma, pred):
    """for each geminate cc in lemma at position i: left context = lemma[max(0,i-3):i]; find in pred (accent removed
    from consideration) the occurrence 'ctx + c' not followed by c, where ctx matches allowing akanye o->a; double c."""
    out = pred
    for i, c in geminates(lemma):
        so = strip(out)
        if c + c in so or c + 'ъ' + c in so or c + 'ь' + c in so: continue
        if any(c + sep + alt in so for sep in ('', 'ь') for alt in PAL.get(c, '')): continue
        ctx = lemma[max(0, i - 3):i]
        # akanye/yat-tolerant regex for the context
        pat = ''.join(('[оа]' if ch == 'о' else '[еѣи]' if ch == 'е' else re.escape(ch)) + ACC + '?' for ch in ctx)
        m = re.search(pat + c + ACC + '?(?!' + c + ')', out)
        if m:
            # insert c after the matched single c (keep any accent following it after the doubled pair)
            e = m.end()
            if out[e - 1] == ACC: out = out[:e - 1] + c + out[e - 1:]
            else: out = out[:e] + c + out[e:]
    return out
if __name__ == '__main__':
    files = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--train_check' in sys.argv:
        tr = pd.read_csv('data/train.csv')
        n = keep = 0; lost = collections.Counter()
        for l, fm in zip(tr.lemma_ru, tr.form_vvz):
            g = geminates(l)
            if not g: continue
            for i, c in g:
                n += 1
                if c + c in strip(fm): keep += 1
                else: lost[c] += 1
        print(f'train: lemma geminates {n}, kept in gold {keep} ({keep/n:.4f}); lost by letter {lost.most_common(6)}')
    recs = []
    for f in files: recs += pickle.load(open(f, 'rb'))
    stats = collections.defaultdict(collections.Counter); new = {}
    for k, r in enumerate(recs):
        new[k] = fix(r['lemma'], r['pred'])
        if new[k] == r['pred']: continue
        was = r['pred'] == r['gold']; now = new[k] == r['gold']
        stats[r['seg']]['changed'] += 1
        if now and not was: stats[r['seg']]['fixed'] += 1
        elif was and not now: stats[r['seg']]['broke'] += 1; stats[r['seg']]['broke_ex'] = (r['lemma'], r['gold'], new[k])
        else: stats[r['seg']]['still_wrong'] += 1
    for seg in ['transfer', 'complete', 'wug', 'unseen']:
        if stats[seg]: print(seg, dict(stats[seg]))
    segs = sorted({r['seg'] for r in recs})
    pb = {s: [r['pred'] for r in recs if r['seg'] == s] for s in segs}
    pn = {s: [new[k] for k, r in enumerate(recs) if r['seg'] == s] for s in segs}
    g = {s: [r['gold'] for r in recs if r['seg'] == s] for s in segs}
    print('before', weighted_score(pb, g)); print('after ', weighted_score(pn, g))
    pickle.dump(new, open('runs/exp/residual-errors/fixC_new.pkl', 'wb'))
