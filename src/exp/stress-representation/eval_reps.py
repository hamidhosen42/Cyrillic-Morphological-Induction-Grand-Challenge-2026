"""Evaluate alternative stress representations for paradigm inference (mask-aware k-medoids).

Every representation maps (form, lemma, pos, dialect) -> a set of compatible classes (bitmask).
Ambiguity (e.g. initial == stem-final for monosyllabic stems, ending -> stem-final fallback when the
ending has no vowel) is handled by summing P over the compatible classes.

Pipeline decision (same as src/candmerge.py base): per (lemma, feats) group
    c* = argmax_c  sum_rows log mass_c(row) + lam * log prior_c
then each row takes its best candidate compatible with c*.

  python3 src/exp/stress-representation/eval_reps.py --reps cur5,3way --lams 2 [--limit N]
"""
import os, sys, pickle, collections, argparse, time, json, re
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))
from validate import load_splits, weighted_score, W, TEST_N
from paradigm import ACC, VOW, stress_idx, n_vow, stress_class
from stemrule import stem_vowels_rule

OUT = os.path.join(ROOT, 'runs', 'exp', 'stress-representation'); os.makedirs(OUT, exist_ok=True)
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
SEGS = ['transfer', 'complete', 'wug', 'unseen']
SHARE = {'transfer': .131, 'complete': .218, 'wug': .398, 'unseen': .254}


# ---------------------------------------------------------------- representations
def bits(*cs):
    m = 0
    for c in cs: m |= 1 << c
    return m


class Rep:
    """name, n classes, mask(form, lemma, pos, dialect) -> bitmask (0 = no accent / unusable)."""
    def __init__(self, name, n, fn):
        self.name, self.n, self.fn = name, n, fn

    def mask(self, form, lemma, pos, dialect):
        si = stress_idx(form)
        if si < 0: return 0
        return self.fn(si, n_vow(form), stem_vowels_rule(lemma, pos, dialect))


def r_cur5(si, nv, ns):
    return bits(0 if si == 0 else int(min(1 + (nv - 1 - si), 4)))

def r_cur_uncapped(si, nv, ns):  # like cur5 but distance uncapped (cap 7 -> 9 classes)
    return bits(0 if si == 0 else int(min(1 + (nv - 1 - si), 8)))

def r_fromstart(si, nv, ns):     # stress index from the start, capped at 6 -> 7 classes
    return bits(int(min(si, 6)))

def r_fromend(si, nv, ns):       # distance from the last vowel, capped at 6 -> 7 classes (no initial flag)
    return bits(int(min(nv - 1 - si, 6)))

def r_rel(si, nv, ns):           # position relative to the stem boundary: si - ns in [-4, +1] -> 6 classes
    return bits(int(np.clip(si - ns, -4, 1)) + 4)

def r_rel_init(si, nv, ns):      # rel + initial flag: initial is its own class (7 classes)
    if si == 0 and ns > 1: return bits(6)
    return bits(int(np.clip(si - ns, -4, 1)) + 4)

# 3-way: I=0, S=1, E=2, M=3 (other). Ambiguity: monosyllabic stem: initial == stem-final;
# zero-vowel ending: 'ending' falls back to stem-final.
def r_3way(si, nv, ns):
    ne = nv - ns
    m = 0
    if si == 0: m |= bits(0)
    if si == ns - 1:
        m |= bits(1)
        if ne <= 0: m |= bits(2)
    if si == ns and ne > 0: m |= bits(2)
    if m == 0: m = bits(3)
    return m

def r_3way_strict(si, nv, ns):   # no ending->stem-final fallback ambiguity
    m = 0
    if si == 0: m |= bits(0)
    if si == ns - 1: m |= bits(1)
    if si == ns and nv - ns > 0: m |= bits(2)
    if m == 0: m = bits(3)
    return m

def r_3way_noamb(si, nv, ns):    # hard labels: initial wins over stem-final; stem-final in zero-ending cells = S
    if si == 0: return bits(0)
    if si == ns - 1: return bits(1)
    if si == ns and nv - ns > 0: return bits(2)
    return bits(3)

def r_3way_fine(si, nv, ns):     # I, S, E, M1 (si == ns-2), M2+ (si <= ns-3), E2+ (si > ns) -> 6 classes
    ne = nv - ns
    m = 0
    if si == 0: m |= bits(0)
    if si == ns - 1:
        m |= bits(1)
        if ne <= 0: m |= bits(2)
    if si == ns and ne > 0: m |= bits(2)
    if m == 0:
        if si == ns - 2: m = bits(3)
        elif si < ns - 2: m = bits(4)
        else: m = bits(5)
    return m

REPS = {
    'cur5': Rep('cur5', 5, r_cur5),
    'cur_uncapped': Rep('cur_uncapped', 9, r_cur_uncapped),
    'fromstart': Rep('fromstart', 7, r_fromstart),
    'fromend': Rep('fromend', 7, r_fromend),
    'rel': Rep('rel', 6, r_rel),
    'rel_init': Rep('rel_init', 7, r_rel_init),
    '3way': Rep('3way', 4, r_3way),
    '3way_strict': Rep('3way_strict', 4, r_3way_strict),
    '3way_noamb': Rep('3way_noamb', 4, r_3way_noamb),
    '3way_fine': Rep('3way_fine', 6, r_3way_fine),
}


# ---------------------------------------------------------------- mask-aware paradigm model
class MaskParadigmModel:
    def __init__(self, rep, K=300, iters=25, seed=0, alpha=1.0):
        self.rep, self.K, self.iters, self.seed, self.alpha = rep, K, iters, seed, alpha
        self.models = {}

    def fit(self, df):
        n = self.rep.n
        rng = np.random.default_rng(self.seed)
        masks = [self.rep.mask(fm, l, p, d) for fm, l, p, d in zip(df.form_vvz, df.lemma_ru, df.pos, df.dialect)]
        df = df.assign(mask=masks)
        df = df[df['mask'] > 0]
        for pos, d in df.groupby('pos'):
            g = d.groupby(['lemma_ru', 'feats'])['mask'].agg(lambda s: collections.Counter(s).most_common(1)[0][0])
            x = g.unstack()
            cells = list(x.columns)
            A = x.fillna(0).values.astype(np.int64)
            L, C = A.shape; K = self.K
            has = A > 0
            # bit membership tensor B[l, j, c] = class c compatible in cell j of lemma l
            B = np.stack([((A >> c) & 1).astype(np.float32) for c in range(n)], -1)   # (L, C, n)
            pop = B.sum(-1); Bw = B / np.maximum(pop, 1)[:, :, None]                   # fractional weights
            init = rng.choice(L, K, replace=False)
            cents = np.zeros((K, C), dtype=np.int64)
            for ki, i in enumerate(init):
                for j in range(C):
                    if A[i, j] > 0:
                        opts = [c for c in range(n) if (A[i, j] >> c) & 1]
                        cents[ki, j] = rng.choice(opts)
                    else:
                        cents[ki, j] = rng.integers(0, n)
            asg = None
            for it in range(self.iters):
                dist = np.zeros((L, K), dtype=np.float32)
                for k in range(K):
                    dist[:, k] = (has & (((A >> cents[k][None, :]) & 1) == 0)).sum(1)
                new = dist.argmin(1)
                if asg is not None and np.array_equal(new, asg): break
                asg = new
                for k in range(K):
                    sel = asg == k
                    if not sel.any(): continue
                    cnt = B[sel].sum(0)                     # (C, n)
                    ok = cnt.sum(1) > 0
                    cents[k, ok] = cnt[ok].argmax(1)
            P = np.full((K, C, n), self.alpha, dtype=np.float64)
            for k in range(K):
                sel = asg == k
                if sel.any(): P[k] += Bw[sel].sum(0)
            P /= P.sum(-1, keepdims=True)
            prior = np.bincount(asg, minlength=K) + 1.0; prior /= prior.sum()
            self.models[pos] = dict(cells=cells, cidx={c: i for i, c in enumerate(cells)}, P=P, prior=prior, logP=np.log(P))
        return self

    def posterior(self, pos, obs):
        """obs: list of (feats, mask). Returns (K,) posterior over paradigms."""
        m = self.models[pos]
        logp = np.log(m['prior']).copy()
        seen = collections.defaultdict(list)
        for f, mk in obs:
            if f in m['cidx'] and mk > 0: seen[f].append(mk)
        n = self.rep.n
        for f, mks in seen.items():
            mk = collections.Counter(mks).most_common(1)[0][0]
            sel = np.array([(mk >> c) & 1 for c in range(n)], dtype=bool)
            logp += np.log(m['P'][:, m['cidx'][f], sel].sum(-1))
        w = np.exp(logp - logp.max()); return w / w.sum()

    def class_dist(self, pos, feats, obs):
        m = self.models[pos]
        if feats not in m['cidx']: return np.full(self.rep.n, 1.0 / self.rep.n)
        w = self.posterior(pos, obs)
        return w @ m['P'][:, m['cidx'][feats], :]


def register_main():
    """Pickles were written by this file run as __main__; make them loadable from other scripts."""
    import __main__
    for name, obj in globals().items():
        if name.startswith('r_') or name in ('Rep', 'MaskParadigmModel', 'bits'):
            setattr(__main__, name, obj)


def get_models(rep, K, seeds, train_rows):
    register_main()
    out = []
    for s in seeds:
        fn = os.path.join(OUT, f'pm_{rep.name}_K{K}_s{s}.pkl')
        if os.path.exists(fn):
            out.append(pickle.load(open(fn, 'rb'))); continue
        t = time.time(); pm = MaskParadigmModel(rep, K=K, seed=s).fit(train_rows)
        pickle.dump(pm, open(fn, 'wb')); print(f'  fit {rep.name} K={K} seed={s} in {time.time() - t:.0f}s', flush=True)
        out.append(pm)
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(rep, seg, df, cpm, known_map, pms, lams, mode='pipeline', decision='class'):
    """Returns dict lam -> (EM, CER) plus diagnostics. mode: 'pipeline' | 'prior' (prior-only class) | 'nn' (lam=0)."""
    rows = list(df.itertuples()); nrows = len(rows); n = rep.n
    # candidate tallies (single model here, but keep generic)
    tallies = []
    for i in range(nrows):
        t = collections.defaultdict(float)
        for mdl in cpm:
            lst = mdl[i]; lp = np.array([c[1] for c in lst]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
            for (s, _), q in zip(lst, pr): t[s] += q / len(cpm)
        tallies.append(t)
    cmask = []; masses = []
    for r, t in zip(rows, tallies):
        mk = {s: rep.mask(s, r.lemma_ru, r.pos, r.dialect) for s in t}
        cmask.append(mk)
        m = np.full(n, 1e-9)
        for s, q in t.items():
            for c in range(n):
                if (mk[s] >> c) & 1: m[c] += q
        masses.append(m)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    # priors per group
    priors = {}
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        dls = {rows[i].dialect for i in idx}
        obs = [(f, rep.mask(fm, lemma, r0.pos, d)) for f, d, fm in known_map.get(lemma, []) if not (f == feats and d in dls)]
        priors[(lemma, feats)] = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
    golds = [r.form_vvz for r in rows]
    out = {}
    from rapidfuzz.distance import Levenshtein as Lev
    # mask-level tables: per row, mass per exact mask
    mmass = [collections.defaultdict(float) for _ in rows]
    for i, t in enumerate(tallies):
        for s, q in t.items(): mmass[i][cmask[i][s]] += q
    for lam in lams:
        preds = []
        for (lemma, feats), idx in groups.items():
            pr = priors[(lemma, feats)]
            if decision == 'mask':
                # decide over exact candidate masks; prior of a mask = sum of prior over its classes
                ms = set()
                for i in idx: ms |= set(mmass[i])
                best_m, best_s = None, -np.inf
                for m in ms:
                    pm = sum(pr[c] for c in range(n) if (m >> c) & 1)
                    nnm = sum(np.log(mmass[i].get(m, 0.0) + 1e-9) for i in idx)
                    if mode == 'prior': s_ = np.log(pm + 1e-9) + (0.0 if all(mmass[i].get(m, 0.0) > 1e-8 for i in idx) else -1e6)
                    elif mode == 'nn': s_ = nnm
                    else: s_ = nnm + lam * np.log(pm + 1e-9)
                    if s_ > best_s: best_m, best_s = m, s_
                for i in idx:
                    best = max(((s, q) for s, q in tallies[i].items() if cmask[i][s] == best_m), key=lambda x: x[1], default=None)
                    if best is None:
                        best = max(((s, q) for s, q in tallies[i].items() if cmask[i][s] & best_m), key=lambda x: x[1], default=None)
                    preds.append((i, best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]))
                continue
            if mode == 'prior':
                # choose the prior-best class that has any candidate mass in the group
                avail = sum(masses[i] > 1e-8 for i in idx) > 0
                sc = np.where(avail, np.log(pr + 1e-9), -1e9)
            elif mode == 'nn':
                sc = sum(np.log(masses[i]) for i in idx)
            else:
                sc = sum(np.log(masses[i]) for i in idx) + lam * np.log(pr + 1e-9)
            c = int(sc.argmax())
            for i in idx:
                best = max(((s, q) for s, q in tallies[i].items() if (cmask[i][s] >> c) & 1), key=lambda x: x[1], default=None)
                preds.append((i, best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]))
        preds.sort(); preds = [p for _, p in preds]
        em = np.mean([p == g for p, g in zip(preds, golds)])
        cer = np.mean([Lev.distance(p, g) / max(1, len(g)) for p, g in zip(preds, golds)])
        pos_acc = np.mean([stress_idx(p) == stress_idx(g) for p, g in zip(preds, golds)])
        out[lam] = (float(em), float(cer), float(pos_acc))
        if mode != 'pipeline': break
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', default='cur5,3way')
    ap.add_argument('--lams', default='2')
    ap.add_argument('--K', type=int, default=300)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--segs', default='transfer,complete,wug,unseen')
    ap.add_argument('--modes', default='pipeline,prior')
    ap.add_argument('--tag', default='')
    ap.add_argument('--decision', default='class', help='class | mask')
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(',')]; seeds = [int(s) for s in args.seeds.split(',')]
    t0 = time.time()
    train_rows, known, val = load_splits()
    ext = pickle.load(open(os.path.join(ROOT, 'runs', 'kdev2', 'val_cands_aligned.pkl'), 'rb'))
    known_map = collections.defaultdict(list)
    for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
        known_map[l].append((f, d, fm))
    print(f'loaded in {time.time() - t0:.0f}s', flush=True)
    results = {}
    for rname in args.reps.split(','):
        rep = REPS[rname]
        pms = get_models(rep, args.K, seeds, train_rows)
        for seg in args.segs.split(','):
            df = val[seg]; cpm = ext[KEY[seg]]
            if args.limit: df = df.head(args.limit); cpm = [m[:args.limit] for m in cpm]
            for mode in args.modes.split(','):
                t = time.time()
                res = evaluate(rep, seg, df, cpm, known_map, pms, lams, mode=mode, decision=args.decision)
                for lam, (em, cer, pa) in res.items():
                    results[f'{rname}|{seg}|{mode}|{lam}|{args.decision}'] = dict(em=em, cer=cer, pos_acc=pa, n=len(df))
                    print(f'{rname:12s} {seg:9s} {mode:8s} {args.decision:5s} lam={lam:<4g} n={len(df):5d} EM={em:.4f} CER={cer:.4f} posacc={pa:.4f} ({time.time() - t:.0f}s)', flush=True)
    fn = os.path.join(OUT, f'results{args.tag}_lim{args.limit}.json')
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(results); json.dump(old, open(fn, 'w'), indent=1)
    print('saved', fn)
