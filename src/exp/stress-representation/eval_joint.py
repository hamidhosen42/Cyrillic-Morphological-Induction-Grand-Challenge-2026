"""Joint per-lemma paradigm inference under a representation (mask-aware).

Posterior over paradigms k for a lemma uses (a) hard observations (known cells, as masks) and
(b) the NN's class mass of every unlabelled row of the lemma (per cell, averaged over dialects):
    log P(k) = log prior[k] + sum_obs log sum_{c in mask} P[k,cell,c] + rw * sum_cells log sum_c P[k,cell,c] mass_c
Then per (lemma, feats) group the usual decision  argmax_c sum_rows log mass_c + lam log P(c | lemma).

Also reports a candidate-free diagnostic: prior-only stress-position accuracy on the gold forms.

  python3 src/exp/stress-representation/eval_joint.py --reps cur5,3way --lams 2 --rws 0,0.5,1
"""
import os, sys, pickle, collections, argparse, time, json
import numpy as np
from rapidfuzz.distance import Levenshtein as Lev
sys.path.insert(0, os.path.dirname(__file__))
from eval_reps import REPS, MaskParadigmModel, get_models, OUT, KEY, ROOT
sys.path.insert(0, os.path.join(ROOT, 'src'))
from validate import load_splits
from paradigm import stress_idx, n_vow
from stemrule import stem_vowels_rule


def joint_eval(rep, seg, df, cpm, known_map, pms, lams, rws, temp=1.0):
    rows = list(df.itertuples()); nrows = len(rows); n = rep.n
    tallies = []
    for i in range(nrows):
        t = collections.defaultdict(float)
        for mdl in cpm:
            lst = mdl[i]; lp = np.array([c[1] for c in lst]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
            for (s, _), q in zip(lst, pr): t[s] += q / len(cpm)
        tallies.append(t)
    cmask, masses = [], []
    for r, t in zip(rows, tallies):
        mk = {s: rep.mask(s, r.lemma_ru, r.pos, r.dialect) for s in t}
        cmask.append(mk)
        m = np.full(n, 1e-9)
        for s, q in t.items():
            for c in range(n):
                if (mk[s] >> c) & 1: m[c] += q
        masses.append(m / m.sum())
    by_lemma = collections.defaultdict(list)
    for i, r in enumerate(rows): by_lemma[r.lemma_ru].append(i)
    golds = [r.form_vvz for r in rows]
    res = {}
    for rw in rws:
        # per (lemma, feats): class distribution given joint posterior
        prior = {}
        for lemma, idx in by_lemma.items():
            pos = rows[idx[0]].pos
            cells = collections.defaultdict(list)
            for i in idx: cells[rows[i].feats].append(i)
            cell_mass = {f: np.mean([masses[i] for i in ii], axis=0) for f, ii in cells.items()}
            dist = {f: np.zeros(n) for f in cells}
            for pm in pms:
                m = pm.models[pos]
                for f in cells:
                    # observations: known forms of the lemma except the same cell in the group's dialects
                    dls = {rows[i].dialect for i in cells[f]}
                    obs = [(ff, rep.mask(fm, lemma, pos, d)) for ff, d, fm in known_map.get(lemma, []) if not (ff == f and d in dls)]
                    logp = np.log(m['prior']).copy()
                    seen = collections.defaultdict(list)
                    for ff, mk in obs:
                        if ff in m['cidx'] and mk > 0: seen[ff].append(mk)
                    for ff, mks in seen.items():
                        mk = collections.Counter(mks).most_common(1)[0][0]
                        sel = np.array([(mk >> c) & 1 for c in range(n)], dtype=bool)
                        logp += np.log(m['P'][:, m['cidx'][ff], sel].sum(-1))
                    if rw > 0:
                        for f2, mm in cell_mass.items():
                            if f2 == f or f2 not in m['cidx']: continue   # exclude the target cell's own NN evidence
                            mm2 = np.power(np.maximum(mm, 1e-9), temp); mm2 /= mm2.sum()
                            logp += rw * np.log((m['P'][:, m['cidx'][f2], :] * mm2[None, :]).sum(1) + 1e-12)
                    w = np.exp(logp - logp.max()); w /= w.sum()
                    dist[f] += (w @ m['P'][:, m['cidx'][f], :]) if f in m['cidx'] else np.full(n, 1.0 / n)
            for f in cells: prior[(lemma, f)] = dist[f] / len(pms)
        for lam in lams:
            preds = [None] * nrows
            for lemma, idx in by_lemma.items():
                cells = collections.defaultdict(list)
                for i in idx: cells[rows[i].feats].append(i)
                for f, ii in cells.items():
                    sc = sum(np.log(masses[i]) for i in ii) + lam * np.log(prior[(lemma, f)] + 1e-9)
                    c = int(sc.argmax())
                    for i in ii:
                        best = max(((s, q) for s, q in tallies[i].items() if (cmask[i][s] >> c) & 1), key=lambda x: x[1], default=None)
                        preds[i] = best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]
            em = np.mean([p == g for p, g in zip(preds, golds)])
            cer = np.mean([Lev.distance(p, g) / max(1, len(g)) for p, g in zip(preds, golds)])
            res[(rw, lam)] = (float(em), float(cer))
    return res


def gold_position_acc(rep, seg, df, known_map, pms):
    """Candidate-free: prior-only class -> position in the gold form; accuracy of the stressed vowel index."""
    rows = list(df.itertuples()); n = rep.n
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    hit = tot = 0
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]; dls = {rows[i].dialect for i in idx}
        obs = [(f, rep.mask(fm, lemma, r0.pos, d)) for f, d, fm in known_map.get(lemma, []) if not (f == feats and d in dls)]
        pr = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        for i in idx:
            g = rows[i].form_vvz; nv = n_vow(g); ns = stem_vowels_rule(lemma, r0.pos, rows[i].dialect)
            # realizable positions in the gold word and their masks; choose the position with max prior mass
            best, bs = None, -1
            for si in range(nv):
                mk = rep.fn(si, nv, ns)
                s = sum(pr[c] for c in range(n) if (mk >> c) & 1)
                if s > bs: best, bs = si, s
            hit += int(best == stress_idx(g)); tot += 1
    return hit / max(1, tot)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', default='cur5,3way')
    ap.add_argument('--lams', default='2')
    ap.add_argument('--rws', default='0,0.5,1')
    ap.add_argument('--segs', default='wug,unseen,complete')
    ap.add_argument('--K', type=int, default=300)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='_joint')
    ap.add_argument('--posacc', action='store_true')
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(',')]; rws = [float(x) for x in args.rws.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    train_rows, known, val = load_splits()
    ext = pickle.load(open(os.path.join(ROOT, 'runs', 'kdev2', 'val_cands_aligned.pkl'), 'rb'))
    known_map = collections.defaultdict(list)
    for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
        known_map[l].append((f, d, fm))
    results = {}
    for rname in args.reps.split(','):
        rep = REPS[rname]; pms = get_models(rep, args.K, seeds, train_rows)
        for seg in args.segs.split(','):
            df = val[seg]; cpm = ext[KEY[seg]]
            if args.limit: df = df.head(args.limit); cpm = [m[:args.limit] for m in cpm]
            if args.posacc:
                t = time.time(); pa = gold_position_acc(rep, seg, df, known_map, pms)
                results[f'{rname}|{seg}|goldpos'] = pa
                print(f'{rname:12s} {seg:9s} prior-only gold-position acc = {pa:.4f} n={len(df)} ({time.time() - t:.0f}s)', flush=True)
            t = time.time(); res = joint_eval(rep, seg, df, cpm, known_map, pms, lams, rws)
            for (rw, lam), (em, cer) in res.items():
                results[f'{rname}|{seg}|joint|rw{rw}|lam{lam}'] = dict(em=em, cer=cer, n=len(df))
                print(f'{rname:12s} {seg:9s} joint rw={rw:<4g} lam={lam:<4g} n={len(df):5d} EM={em:.4f} CER={cer:.4f} ({time.time() - t:.0f}s)', flush=True)
    fn = os.path.join(OUT, f'results{args.tag}_lim{args.limit}.json')
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(results); json.dump(old, open(fn, 'w'), indent=1); print('saved', fn)
