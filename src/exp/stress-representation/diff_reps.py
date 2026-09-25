"""Row-level disagreement analysis between two representations' pipeline decisions (same lam).
  python3 src/exp/stress-representation/diff_reps.py --a cur5 --b 3way --seg transfer --lam 2 [--show 30]
"""
import os, sys, pickle, collections, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from eval_reps import REPS, get_models, OUT, KEY, ROOT
sys.path.insert(0, os.path.join(ROOT, 'src'))
from validate import load_splits
from paradigm import stress_idx, n_vow, ACC
from stemrule import stem_vowels_rule


def decide(rep, df, cpm, known_map, pms, lam):
    rows = list(df.itertuples()); n = rep.n
    tallies = []
    for i in range(len(rows)):
        t = collections.defaultdict(float)
        for mdl in cpm:
            lst = mdl[i]; lp = np.array([c[1] for c in lst]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
            for (s, _), q in zip(lst, pr): t[s] += q / len(cpm)
        tallies.append(t)
    cmask, masses = [], []
    for r, t in zip(rows, tallies):
        mk = {s: rep.mask(s, r.lemma_ru, r.pos, r.dialect) for s in t}; cmask.append(mk)
        m = np.full(n, 1e-9)
        for s, q in t.items():
            for c in range(n):
                if (mk[s] >> c) & 1: m[c] += q
        masses.append(m)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    preds = [None] * len(rows); info = [None] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]; dls = {rows[i].dialect for i in idx}
        obs = [(f, rep.mask(fm, lemma, r0.pos, d)) for f, d, fm in known_map.get(lemma, []) if not (f == feats and d in dls)]
        pr = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        sc = sum(np.log(masses[i]) for i in idx) + lam * np.log(pr + 1e-9); c = int(sc.argmax())
        for i in idx:
            best = max(((s, q) for s, q in tallies[i].items() if (cmask[i][s] >> c) & 1), key=lambda x: x[1], default=None)
            preds[i] = best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]
            info[i] = (c, np.round(pr, 2), np.round(masses[i] / masses[i].sum(), 2))
    return preds, info, tallies


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='cur5'); ap.add_argument('--b', default='3way')
    ap.add_argument('--seg', default='transfer'); ap.add_argument('--lam', type=float, default=2.0)
    ap.add_argument('--show', type=int, default=30); ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    train_rows, known, val = load_splits()
    ext = pickle.load(open(os.path.join(ROOT, 'runs', 'kdev2', 'val_cands_aligned.pkl'), 'rb'))
    known_map = collections.defaultdict(list)
    for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
        known_map[l].append((f, d, fm))
    df = val[args.seg]; cpm = ext[KEY[args.seg]]
    if args.limit: df = df.head(args.limit); cpm = [m[:args.limit] for m in cpm]
    ra, rb = REPS[args.a], REPS[args.b]
    pa, ia, tal = decide(ra, df, cpm, known_map, get_models(ra, 300, [0, 1, 2], train_rows), args.lam)
    pb, ib, _ = decide(rb, df, cpm, known_map, get_models(rb, 300, [0, 1, 2], train_rows), args.lam)
    rows = list(df.itertuples())
    cnt = collections.Counter(); shown = 0
    for i, r in enumerate(rows):
        g = r.form_vvz
        if pa[i] == pb[i]: continue
        key = ('A_right' if pa[i] == g else 'A_wrong', 'B_right' if pb[i] == g else 'B_wrong')
        cnt[key] += 1
        # same position?
        cnt['samepos' if stress_idx(pa[i]) == stress_idx(pb[i]) else 'diffpos'] += 1
        if shown < args.show and pa[i] != pb[i] and (pa[i] == g or pb[i] == g):
            shown += 1
            ns = stem_vowels_rule(r.lemma_ru, r.pos, r.dialect)
            same = [(f, d, fm) for f, d, fm in known_map.get(r.lemma_ru, []) if f == r.feats]
            top = sorted(tal[i].items(), key=lambda x: -x[1])[:4]
            print(f'{r.lemma_ru} {r.feats} {r.dialect} ns={ns} gold={g} | A={pa[i]} {ia[i][0]} | B={pb[i]} {ib[i][0]} | samecell={same} | cands={[(s, round(q, 2)) for s, q in top]}')
            print(f'    priorA={ia[i][1]} massA={ia[i][2]}  priorB={ib[i][1]} massB={ib[i][2]}')
    print(args.seg, 'n=%d' % len(rows), 'EM A=%.4f B=%.4f' % (np.mean([p == r.form_vvz for p, r in zip(pa, rows)]), np.mean([p == r.form_vvz for p, r in zip(pb, rows)])), dict(cnt))
