"""Exhaustive tuning of the decision rule on the holdout (no new models).

Loads validate.load_splits() + runs/kdev2/val_cands_aligned.pkl once, precomputes per-row
candidate tables (forms, logprobs, stress classes, EM/CER vs gold, pairwise CER for MBR,
length-prior log-probs, segmental-equivalence) and paradigm class priors (K in {300,600},
seeds averaged), then evaluates many decision variants with numpy.

Per-segment objective  0.9*EM + 0.1*(1-CER)  is additive across segments (validate.weighted_score
is a weighted mean of it), so segment parameters are tuned independently.

  python src/exp/decision-tuning/tune.py [--limit N] [--Ks 300,600,600s6] [--cv]
  (a K entry like 600s6 = K=600 averaged over seeds 0..5; plain = seeds 0,1,2)
"""
import os, sys, pickle, collections, itertools, time, argparse, json
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from validate import load_splits, weighted_score, W, TEST_N
from paradigm import ParadigmModel, stress_class, NC
from rapidfuzz.distance import Levenshtein as Lev

OUT = os.path.join(ROOT, 'runs', 'exp', 'decision-tuning'); os.makedirs(OUT, exist_ok=True)
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
SEGS = ['transfer', 'complete', 'wug', 'unseen']
ACC = '́'
POSI = {'N': 0, 'V': 1, 'ADJ': 2}

ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--cands', default=os.path.join(ROOT, 'runs', 'kdev2', 'val_cands_aligned.pkl'))
ap.add_argument('--Ks', default='300,600')
ap.add_argument('--cv', action='store_true', help='2-fold lemma CV of the whole staged procedure')
ap.add_argument('--tag', default='v3')
ap.add_argument('--no_stages', action='store_true')
args = ap.parse_args()
Ks = args.Ks.split(',')

def k_seeds(k):
    if 's' in k: K, ns = k.split('s'); return int(K), list(range(int(ns)))
    return int(k), [0, 1, 2]

t0 = time.time()
train_rows, known, val = load_splits()
ext = pickle.load(open(args.cands, 'rb'))
known_map = collections.defaultdict(list)
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map[l].append((f, d, fm))
print(f'loaded splits+cands in {time.time()-t0:.0f}s', flush=True)

def get_pms(k):
    K, seeds = k_seeds(k); pms = []
    for s in seeds:
        fn = os.path.join(OUT, f'pm_K{K}_s{s}.pkl')
        if s in (0, 1, 2) and not os.path.exists(fn) and os.path.exists(os.path.join(OUT, f'pms_K{K}.pkl')):
            for i, pm in enumerate(pickle.load(open(os.path.join(OUT, f'pms_K{K}.pkl'), 'rb'))):
                pickle.dump(pm, open(os.path.join(OUT, f'pm_K{K}_s{i}.pkl'), 'wb'))
        if os.path.exists(fn):
            pms.append(pickle.load(open(fn, 'rb'))); continue
        t = time.time(); pm = ParadigmModel(K=K, seed=s).fit(train_rows); pickle.dump(pm, open(fn, 'wb')); pms.append(pm)
        print(f'fit ParadigmModel K={K} seed={s} in {time.time()-t:.0f}s', flush=True)
    return pms

# length prior: P(len(form)-len(lemma) | pos, feats, dialect), backoff (pos, feats)
lencnt = collections.defaultdict(collections.Counter)
for l, p, f, d, fm in zip(known.lemma_ru, known.pos, known.feats, known.dialect, known.form_vvz):
    lencnt[(p, f, d)][len(fm.replace(ACC, '')) - len(l)] += 1
lencnt2 = collections.defaultdict(collections.Counter)
for (p, f, d), c in lencnt.items(): lencnt2[(p, f)].update(c)
def loglen(p, f, d, delta):
    c = lencnt.get((p, f, d)) or lencnt2.get((p, f)) or collections.Counter()
    tot = sum(c.values())
    return np.log((c.get(delta, 0) + 0.5) / (tot + 0.5 * 31))

class Seg: pass

def build(seg):
    df = val[seg]; cpm = ext[KEY[seg]]
    if args.limit:
        df = df.head(args.limit); cpm = [m[:args.limit] for m in cpm]
    rows = list(df.itertuples()); n = len(rows); nm = len(cpm)
    S = Seg(); S.name = seg; S.n = n; S.df = df
    S.gold = list(df.form_vvz); S.lemma = list(df.lemma_ru)
    S.pos = np.array([POSI[r.pos] for r in rows])
    forms = []; lps = []
    for i in range(n):
        u = {}
        for m in range(nm):
            for s, lp in cpm[m][i]:
                if s not in u: u[s] = [-np.inf] * nm
                u[s][m] = max(u[s][m], lp)
        forms.append(list(u)); lps.append([u[s] for s in u])
    Bm = max(map(len, forms)); S.B = Bm
    S.forms = np.full((n, Bm), '', dtype=object)
    S.lp = np.full((nm, n, Bm), -np.inf)
    S.cls = np.full((n, Bm), -1, dtype=int)
    S.em = np.zeros((n, Bm)); S.cer = np.ones((n, Bm))
    S.loglen = np.zeros((n, Bm)); S.D = np.zeros((n, Bm, Bm)); S.valid = np.zeros((n, Bm), bool)
    for i, r in enumerate(rows):
        g = S.gold[i]
        for j, s in enumerate(forms[i]):
            S.forms[i, j] = s; S.valid[i, j] = True
            for m in range(nm): S.lp[m, i, j] = lps[i][j][m]
            S.cls[i, j] = stress_class(s)
            S.em[i, j] = float(s == g); S.cer[i, j] = Lev.distance(s, g) / max(1, len(g))
            S.loglen[i, j] = loglen(r.pos, r.feats, r.dialect, len(s.replace(ACC, '')) - len(r.lemma_ru))
        fl = forms[i]
        for a in range(len(fl)):
            for b in range(len(fl)):
                S.D[i, a, b] = Lev.distance(fl[a], fl[b]) / max(1, len(fl[b]))
    S.OH = np.zeros((n, Bm, NC))
    for c in range(NC): S.OH[:, :, c] = (S.cls == c)
    strip = np.vectorize(lambda x: x.replace(ACC, ''))(S.forms)
    S.M = (strip[:, :, None] == strip[:, None, :]) & S.valid[:, :, None] & S.valid[:, None, :]
    gk = {}; S.gid = np.zeros(n, int)
    for i, r in enumerate(rows):
        S.gid[i] = gk.setdefault((r.lemma_ru, r.feats), len(gk))
    S.ng = len(gk); S.gsize = np.bincount(S.gid, minlength=S.ng)
    S.first = np.zeros(S.ng, int)
    for i in range(n - 1, -1, -1): S.first[S.gid[i]] = i
    S.gcls = np.array([stress_class(g) for g in S.gold])
    S.samecell = np.full(n, -1)
    for i, r in enumerate(rows):
        cs = [stress_class(fm) for f, d, fm in known_map.get(r.lemma_ru, []) if f == r.feats and d != r.dialect]
        cs = [c for c in cs if c >= 0]
        if cs: S.samecell[i] = collections.Counter(cs).most_common(1)[0][0]
    S.prior = {}
    for k in Ks:
        if k.startswith('knn'):
            S.prior[k] = np.load(os.path.join(OUT, f'prior_{seg}_{k}_lim{args.limit}.npy')); continue
        K, seeds = k_seeds(k)
        fn = os.path.join(OUT, f'prior_{seg}_K{K}_lim{args.limit}.npy' if seeds == [0, 1, 2] else f'prior_{seg}_K{K}_s{len(seeds)}_lim{args.limit}.npy')
        if os.path.exists(fn):
            S.prior[k] = np.load(fn); continue
        pms = get_pms(k); pr = np.zeros((n, NC)); t = time.time(); cache = {}
        for i, r in enumerate(rows):
            key = (r.lemma_ru, r.feats, r.dialect)
            if key not in cache:
                obs = [(f, fm) for f, d, fm in known_map.get(r.lemma_ru, []) if not (f == r.feats and d == r.dialect)]
                cache[key] = sum(pm.class_dist(r.pos, r.feats, obs) for pm in pms) / len(pms)
            pr[i] = cache[key]
        np.save(fn, pr); S.prior[k] = pr; print(f'  priors {seg} {k} in {time.time()-t:.0f}s', flush=True)
    # lemma ids and a fixed lemma half-split (for CV)
    lem = np.array(S.lemma); ul = np.unique(lem); rng2 = np.random.default_rng(1); rng2.shuffle(ul)
    S.lid = np.array([{l: i for i, l in enumerate(ul)}[l] for l in lem]); S.nlem = len(ul)
    S.h1 = np.isin(lem, ul[:len(ul) // 2])
    return S

t = time.time(); SEG = {s: build(s) for s in SEGS}; print(f'built tables in {time.time()-t:.0f}s', flush=True)

# ---------------- decision rule ----------------
DEFAULT = dict(K=Ks[0], lam=2.0, joint=True, T=1.0, agg='sum', gnorm='sum', sel='hard', mu=0.0, beta=0.0, lam_pos=None,
               eps=0.0, bias=None, fallback='argmax')

def decide(S, cfg):
    c = dict(DEFAULT); c.update(cfg)
    n, Bm = S.n, S.B
    lp = S.lp / c['T']
    fin = np.isfinite(lp)
    mx = np.where(fin.any(2, keepdims=True), np.max(np.where(fin, lp, -1e30), axis=2, keepdims=True), 0)
    pr = np.where(fin, np.exp(lp - mx), 0.0)
    pr = pr / np.maximum(pr.sum(2, keepdims=True), 1e-30)
    tally = pr.mean(0)
    if c['agg'] == 'sum': mass = np.einsum('nk,nkc->nc', tally, S.OH)
    else: mass = (tally[:, :, None] * S.OH).max(1)
    logmass = np.log(mass + 1e-9)
    prior = S.prior[c['K']] if c['K'] != 'both' else sum(S.prior[k] for k in Ks) / len(Ks)
    if c['eps']: prior = (1 - c['eps']) * prior + c['eps'] / NC
    logprior = np.log(prior + 1e-9)
    lam = np.full(n, float(c['lam']))
    if c['lam_pos'] is not None:
        for p, v in c['lam_pos'].items(): lam[S.pos == POSI[p]] = v
    if c['beta']:
        bt = np.where(S.samecell[:, None] == np.arange(NC)[None, :], np.log(0.98), np.log(0.02 / 4)) * c['beta']
        bt[S.samecell < 0] = 0
    else: bt = 0.0
    if c['joint']:
        G = np.zeros((S.ng, NC)); np.add.at(G, S.gid, logmass)
        sc = np.ones(S.ng) if c['gnorm'] == 'sum' else (1.0 / S.gsize if c['gnorm'] == 'mean' else 1.0 / np.sqrt(S.gsize))
        G = G * sc[:, None]
        Sg = G + lam[S.first][:, None] * logprior[S.first] + (bt[S.first] if c['beta'] else 0)
        Srow = Sg[S.gid]
        logw = Srow - logmass * sc[S.gid][:, None]
    else:
        logw = lam[:, None] * logprior + bt
        Srow = logmass + logw
    if c['bias'] is not None:
        Srow = Srow + np.asarray(c['bias'])[None, :]; logw = logw + np.asarray(c['bias'])[None, :]
    cstar = Srow.argmax(1)
    if c['fallback'] == 'next':
        present = S.OH.any(1)
        cstar = np.where(present[np.arange(n), cstar], cstar, np.where(present, Srow, -np.inf).argmax(1))
    selscore = np.log(tally + 1e-12) + c['mu'] * S.loglen
    selscore[~S.valid] = -np.inf
    if c['sel'] in ('hard', 'hard_segmarg'):
        m = (S.cls == cstar[:, None])
        if c['sel'] == 'hard_segmarg':
            pseg = np.einsum('nb,nab->na', tally, S.M)
            selscore = np.log(pseg + 1e-12) + 1e-6 * np.log(tally + 1e-12) + c['mu'] * S.loglen
            selscore[~S.valid] = -np.inf
        sc_ = np.where(m, selscore, -np.inf)
        has = m.any(1)
        idx = np.where(has, sc_.argmax(1), np.where(S.cls >= 0, selscore, -np.inf).argmax(1))
        noacc = ~(S.cls >= 0).any(1)
        if noacc.any(): idx[noacc] = selscore[noacc].argmax(1)
        return idx
    lw = np.take_along_axis(logw, np.maximum(S.cls, 0), axis=1)
    lw[S.cls < 0] = -50.0
    logq = selscore + lw
    if c['sel'] == 'soft':
        return logq.argmax(1)
    if c['sel'] == 'prodmarg':
        pseg = np.einsum('nb,nab->na', tally, S.M)
        lpc = Srow - Srow.max(1, keepdims=True); lpc = lpc - np.log(np.exp(lpc).sum(1, keepdims=True))
        sc_ = np.log(pseg + 1e-12) + np.take_along_axis(lpc, np.maximum(S.cls, 0), axis=1) + c['mu'] * S.loglen
        sc_[(S.cls < 0) | ~S.valid] = -np.inf
        return sc_.argmax(1)
    if c['sel'] == 'mbr_hard':
        m = (S.cls == cstar[:, None]); has = m.any(1)
        logq = np.where(m | ~has[:, None], logq, -np.inf)
    q = np.exp(logq - logq.max(1, keepdims=True)); q = q / q.sum(1, keepdims=True)
    U = 0.9 * q - 0.1 * np.einsum('nb,nab->na', q, S.D)
    U[~S.valid] = -np.inf
    return U.argmax(1)

FIT_MASK = {s: None for s in SEGS}   # rows used by the objective during selection (None = all)

def rowobj(S, idx):
    ar = np.arange(S.n); return 0.9 * S.em[ar, idx] - 0.1 * S.cer[ar, idx]

def seg_obj(S, idx, mask=None):
    ar = np.arange(S.n); m = FIT_MASK[S.name] if mask is None else mask
    em = S.em[ar, idx]; cer = S.cer[ar, idx]
    if m is not None: em = em[m]; cer = cer[m]
    return 0.9 * em.mean() + 0.1 * (1 - cer.mean()), em.mean(), cer.mean()

def show(cfgs, label='', mask=None):
    preds = {}; golds = {}
    for s in SEGS:
        S = SEG[s]; idx = decide(S, cfgs[s]); m = mask[s] if mask is not None else np.ones(S.n, bool)
        preds[s] = [S.forms[i, j] for i, j in enumerate(idx) if m[i]]; golds[s] = [g for i, g in enumerate(S.gold) if m[i]]
    sc, per2 = weighted_score(preds, golds)
    print(f'{label:44s} score {sc:.5f} | ' + ' '.join(f'{s}: EM {per2[s][0]:.4f} CER {per2[s][1]:.4f}' for s in SEGS), flush=True)
    return sc, per2, preds

LAMS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 10, 1000]
VERBOSE = True
RES = {'stages': {}}

def sweep(name, grid_fn, base):
    if VERBOSE: print(f'\n=== STAGE {name}', flush=True)
    best = {}
    for s in SEGS:
        S = SEG[s]; rowsout = []
        for cfg in grid_fn(s, base[s]):
            o, em, cer = seg_obj(S, decide(S, cfg)); rowsout.append((o, em, cer, cfg))
        rowsout.sort(key=lambda x: -x[0])
        b = rowsout[0]; b0 = seg_obj(S, decide(S, base[s]))
        best[s] = b[3] if b[0] > b0[0] + 1e-12 else base[s]     # keep base on ties (fewer changes)
        if VERBOSE:
            print(f'{s:9s} base obj {b0[0]:.5f} (EM {b0[1]:.4f}) -> best obj {b[0]:.5f} (EM {b[1]:.4f} CER {b[2]:.4f}) d_obj={b[0]-b0[0]:+.5f} cfg={ {k:v for k,v in b[3].items() if DEFAULT.get(k)!=v} }')
            for o, em, cer, cfg in rowsout[:6]:
                print(f'     {o:.5f} EM {em:.4f} CER {cer:.4f} { {k:v for k,v in cfg.items() if DEFAULT.get(k)!=v} }')
            RES['stages'].setdefault(name, {})[s] = [(o, em, cer, {k: (v if not isinstance(v, dict) else dict(v)) for k, v in cfg.items()}) for o, em, cer, cfg in rowsout]
    return best

def gA(s, b): return [dict(b, K=K, joint=j, lam=l) for K in Ks for j in (True, False) for l in LAMS]
def gB(s, b): return [dict(b, T=T, lam=l) for T in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0) for l in LAMS]
def gC(s, b): return [dict(b, agg=a, gnorm=g, lam=l) for a in ('sum', 'max') for g in ('sum', 'mean', 'sqrt') for l in LAMS]
def gC2(s, b): return [dict(b, K=K, eps=e, lam=l) for K in (Ks + ['both'] if len(Ks) > 1 else Ks) for e in (0, 0.02, 0.05, 0.1, 0.2) for l in LAMS]
def gD(s, b): return [dict(b, sel=se, mu=mu) for se in ('hard', 'hard_segmarg', 'prodmarg', 'soft', 'mbr', 'mbr_hard') for mu in (0, 0.15, 0.3, 0.6, 1.0)]
def gE(s, b): return [dict(b, beta=bt) for bt in (0, 0.5, 1, 2, 4)]
def gG(s, b): return [dict(b, fallback=fb) for fb in ('argmax', 'next')]
def gF(s, b):
    poss = sorted(set(SEG[s].pos)); names = [p for p, i in POSI.items() if i in poss]
    return [dict(b, lam_pos=dict(zip(names, combo))) for combo in itertools.product([0.5, 1, 1.5, 2, 3, 4, 6, 10], repeat=len(names))]

def fit_bias(s, base):
    S = SEG[s]; b = np.zeros(NC); cur = seg_obj(S, decide(S, base))[0]
    for _ in range(2):
        for ci in range(NC):
            for v in (-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5):
                bb = b.copy(); bb[ci] = v; o = seg_obj(S, decide(S, dict(base, bias=bb)))[0]
                if o > cur + 1e-9: cur, b = o, bb
    return b

def staged(base, with_pos=True, with_bias=True):
    """runs the whole staged selection (objective restricted by FIT_MASK); returns dict of stage -> per-seg cfg"""
    out = {}
    b = sweep('A: K x joint x lam (T=1,sum,hard)', gA, base); out['A'] = b
    b = sweep('B: temperature x lam', gB, b); out['B'] = b
    b = sweep('C: mass agg x group norm x lam', gC, b); out['C'] = b
    b = sweep('C2: prior eps-mix x K(both) x lam', gC2, b); out['C2'] = b
    b = sweep('D: selection rule x length prior mu', gD, b); out['D'] = b
    b = sweep('E: same-cell class boost beta', gE, b); out['E'] = b
    b = sweep('G1: fallback when class absent from beam', gG, b); out['G1'] = b
    out['nobias'] = b
    if with_bias:
        if VERBOSE: print('\n=== STAGE G2: per-class additive bias on class score (coordinate ascent, 2 passes)')
        bb = {}
        for s in SEGS:
            bias = fit_bias(s, b[s]); bb[s] = dict(b[s], bias=bias.tolist())
            if VERBOSE:
                o0 = seg_obj(SEG[s], decide(SEG[s], b[s])); o1 = seg_obj(SEG[s], decide(SEG[s], bb[s]))
                print(f'{s:9s} obj {o0[0]:.5f} -> {o1[0]:.5f} (EM {o1[1]:.4f} CER {o1[2]:.4f}) d_obj={o1[0]-o0[0]:+.5f} bias={bias.tolist()}')
        b = bb; out['G2'] = b
    if with_pos:
        b = sweep('F: per-POS lam', gF, b); out['F'] = b
    out['final'] = b
    return out

# ---------------- diagnostics ----------------
print('\n=== DIAGNOSTICS (n per segment: ' + ', '.join(f'{s}={SEG[s].n} rows/{SEG[s].nlem} lemmas' for s in SEGS) + ')')
for s in SEGS:
    S = SEG[s]; ar = np.arange(S.n)
    orac = S.em.max(1).mean()
    nn = decide(S, dict(sel='hard', lam=0.0, joint=False))
    m = (S.cls == S.gcls[:, None]); tal = np.exp(S.lp[0]); tal[~S.valid] = 0
    sc = np.where(m, tal, -np.inf); has = m.any(1)
    cor = np.where(has, sc.argmax(1), tal.argmax(1)); orc = S.em[ar, cor].mean()
    base = decide(S, {}); bcls = S.cls[ar, base]
    print(f'{s:9s} oracle(gold in beam) {orac:.4f} | oracle-class then argmax {orc:.4f} | NN-only EM {S.em[ar,nn].mean():.4f} | baseline EM {S.em[ar,base].mean():.4f} class-acc {np.mean(bcls==S.gcls):.4f}'
          f' | baseline errors: stress-wrong {np.mean((bcls!=S.gcls))*100:.1f}% of rows, segmental-only {np.mean((bcls==S.gcls)&(S.em[ar,base]==0))*100:.1f}% of rows')

base_cfg = {s: dict(DEFAULT) for s in SEGS}
BASE, BASE_PER, _ = show(base_cfg, f'BASELINE lam=2 joint K={Ks[0]}')
RES['baseline'] = BASE; RES['baseline_per'] = BASE_PER

# single-factor comparisons at lam=2 joint (for the report)
print('\n=== SINGLE-FACTOR comparisons vs baseline (per-segment obj = 0.9EM+0.1(1-CER); d = new - base)')
for name, cfg in [('K=300 per-row', dict(joint=False)), ('K=600', dict(K='600')), ('K=600 per-row', dict(K='600', joint=False)),
                  ('K=both', dict(K='both')), ('mu=0.3', dict(mu=0.3)), ('mu=0.6', dict(mu=0.6)), ('mu=1.0', dict(mu=1.0)),
                  ('K=600 mu=0.6', dict(K='600', mu=0.6)), ('K=600 per-row mu=0.6', dict(K='600', joint=False, mu=0.6)),
                  ('lam=1', dict(lam=1)), ('lam=3', dict(lam=3)), ('lam=4', dict(lam=4)), ('prior only (lam=1000)', dict(lam=1000)),
                  ('sel=mbr', dict(sel='mbr')), ('sel=soft', dict(sel='soft')), ('sel=hard_segmarg', dict(sel='hard_segmarg')), ('sel=prodmarg', dict(sel='prodmarg')),
                  ('fallback=next', dict(fallback='next')), ('T=0.7', dict(T=0.7)), ('T=1.5', dict(T=1.5))] + \
                 ([('K=600s6', dict(K='600s6')), ('K=600s6 mu=0.6', dict(K='600s6', mu=0.6)), ('K=600s6 per-row mu=0.6', dict(K='600s6', joint=False, mu=0.6))] if '600s6' in Ks else []) + \
                 [(f'K={k}{sfx}', dict(K=k, **extra)) for k in Ks if k.startswith('knn') for sfx, extra in (('', {}), (' mu=0.6', dict(mu=0.6)), (' per-row mu=0.6', dict(joint=False, mu=0.6)), (' lam=1', dict(lam=1)), (' lam=4', dict(lam=4)))]:
    if 'K' in cfg and cfg['K'] not in Ks + ['both']: continue
    line = f'{name:26s}'
    for s in SEGS:
        S = SEG[s]; o0 = seg_obj(S, decide(S, base_cfg[s])); o1 = seg_obj(S, decide(S, dict(base_cfg[s], **cfg)))
        line += f' | {s} d_obj {o1[0]-o0[0]:+.5f} (EM {o1[1]:.4f})'
    print(line, flush=True)

# ---------------- recommended low-parameter configs (3 discrete choices, signs consistent across segments) ----------------
print('\n=== RECOMMENDED configs (few parameters) vs baseline, with lemma-bootstrap CI and per-lemma-half consistency')
rng = np.random.default_rng(0)
K6 = '600' if '600' in Ks else Ks[0]
RECS = {'R1: mu=0.6 only': {s: dict(DEFAULT, mu=0.6) for s in SEGS},
        'R2: mu=0.6 + per-row for unseen': {s: dict(DEFAULT, mu=0.6, joint=(s != 'unseen')) for s in SEGS},
        'R3: K=600 + mu=0.6': {s: dict(DEFAULT, K=K6, mu=0.6) for s in SEGS},
        'R4: K=600 + mu=0.6 + per-row unseen': {s: dict(DEFAULT, K=K6, mu=0.6, joint=(s != 'unseen')) for s in SEGS},
        'R5: R4 + lam=10 transfer, lam=4 complete': {s: dict(DEFAULT, K=K6, mu=0.6, joint=(s != 'unseen'), lam={'transfer': 10, 'complete': 4}.get(s, 2)) for s in SEGS}}
for name, cfgs in RECS.items():
    sc, per, _ = show(cfgs, name)
    line = '      d_obj per seg:'; gain = 0.0
    for s in SEGS:
        S = SEG[s]; d = rowobj(S, decide(S, cfgs[s])) - rowobj(S, decide(S, base_cfg[s]))
        per_l = np.bincount(S.lid, weights=d, minlength=S.nlem); cnt = np.bincount(S.lid, minlength=S.nlem)
        bs = np.array([per_l[p].sum() / cnt[p].sum() for p in (rng.integers(0, S.nlem, S.nlem) for _ in range(1000))])
        line += f' {s} {d.mean():+.5f} [{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}] (h1 {d[S.h1].mean():+.5f} / h2 {d[~S.h1].mean():+.5f})'
    print(line + f' | score delta {sc-BASE:+.5f}', flush=True)
    RES.setdefault('recommended', {})[name] = sc
if args.no_stages:
    json.dump(RES, open(os.path.join(OUT, f'results_{args.tag}_lim{args.limit}.json'), 'w'), indent=1, default=str); sys.exit(0)

# ---------------- full in-sample staged tuning ----------------
FIT_MASK = {s: None for s in SEGS}
ST = staged(base_cfg)
FINAL, FINAL_PER, _ = show(ST['final'], 'FINAL (all stages, in-sample)')
show(ST['nobias'], 'FINAL without class bias and per-POS lam')
show(ST['G2'], 'FINAL without per-POS lam')
RES.update(final=FINAL, final_per=FINAL_PER, final_cfg={s: {k: (v if not isinstance(v, dict) else dict(v)) for k, v in ST['final'][s].items()} for s in SEGS},
           nobias_cfg={s: dict(ST['nobias'][s]) for s in SEGS})
# bootstrap over lemmas of in-sample delta (optimistic: config selected on the same rows)
print('\n=== BOOTSTRAP over lemmas (1000 resamples) of IN-SAMPLE delta obj (final - baseline); optimistic')
rng = np.random.default_rng(0)
for key in ('nobias', 'final'):
    for s in SEGS:
        S = SEG[s]; d = rowobj(S, decide(S, ST[key][s])) - rowobj(S, decide(S, base_cfg[s]))
        per_l = np.bincount(S.lid, weights=d, minlength=S.nlem); cnt = np.bincount(S.lid, minlength=S.nlem)
        bs = np.array([per_l[p].sum() / cnt[p].sum() for p in (rng.integers(0, S.nlem, S.nlem) for _ in range(1000))])
        print(f'{key:7s} {s:9s} d_obj {d.mean():+.5f}  95% CI [{np.percentile(bs,2.5):+.5f}, {np.percentile(bs,97.5):+.5f}]  lemmas={S.nlem}')

# ---------------- 2-fold lemma CV of the whole staged procedure ----------------
if args.cv:
    print('\n=== 2-FOLD LEMMA CV of the ENTIRE staged procedure (select on one lemma-half, score on the other); honest estimate')
    VERBOSE = False
    oof = {k: {s: np.zeros(SEG[s].n) for s in SEGS} for k in ('A', 'D', 'nobias', 'G2', 'final')}
    for fold in (0, 1):
        FIT_MASK = {s: (SEG[s].h1 if fold == 0 else ~SEG[s].h1) for s in SEGS}
        st = staged(base_cfg)
        for k in oof:
            for s in SEGS:
                S = SEG[s]; ev = ~FIT_MASK[s]; oof[k][s][ev] = rowobj(S, decide(S, st[k][s]))[ev]
        print(f'fold {fold} selected (final): ' + '; '.join(f"{s}: { {kk:v for kk,v in st['final'][s].items() if DEFAULT.get(kk)!=v} }" for s in SEGS), flush=True)
        print(f'fold {fold} selected (nobias): ' + '; '.join(f"{s}: { {kk:v for kk,v in st['nobias'][s].items() if DEFAULT.get(kk)!=v} }" for s in SEGS), flush=True)
    VERBOSE = True; FIT_MASK = {s: None for s in SEGS}
    share = {s: W[s] * TEST_N[s] for s in SEGS}; tot = sum(share.values())
    print('\nOut-of-fold delta obj vs baseline (rows weighted like the test; LB-score delta = sum_seg share * d_obj):')
    for k in ('A', 'D', 'nobias', 'G2', 'final'):
        line = f'after {k:7s}'; gain = 0.0
        for s in SEGS:
            S = SEG[s]; d = oof[k][s] - rowobj(S, decide(S, base_cfg[s])); gain += share[s] / tot * d.mean()
            per_l = np.bincount(S.lid, weights=d, minlength=S.nlem); cnt = np.bincount(S.lid, minlength=S.nlem)
            bs = np.array([per_l[p].sum() / cnt[p].sum() for p in (rng.integers(0, S.nlem, S.nlem) for _ in range(500))])
            line += f' | {s} {d.mean():+.5f} [{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}]'
        print(line + f' | est. score delta {gain:+.5f}', flush=True)
        RES.setdefault('cv', {})[k] = gain

json.dump(RES, open(os.path.join(OUT, f'results_{args.tag}_lim{args.limit}.json'), 'w'), indent=1, default=str)
print(f'\nsaved results_{args.tag}_lim{args.limit}.json; total time {time.time()-t0:.0f}s')
