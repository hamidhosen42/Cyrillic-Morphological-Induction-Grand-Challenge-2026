"""Final decision pipeline over beam candidates.

Per segment (transfer / complete / wug / unseen) choose:
  rep   : stress representation used for the class decision
          'cur5' = src/paradigm.stress_class (initial / distance from the last vowel)
          '3way' = initial / stem-final / ending relative to the lemma stem (src/exp/stress-representation)
  lam   : weight of the paradigm prior vs the candidate class mass
  joint : decide one class per (lemma, feats) group (True) or per row (False)
Then, inside the chosen class, pick the candidate maximising log(mass) + mu * log P(length delta).

  python3 src/final.py --mode dev                                  # holdout validation (kdev2 candidates)
  python3 src/final.py --mode full --out runs/sub_final.csv        # test submission (v1v3 candidates)
"""
import argparse, collections, json, os, pickle, sys, time
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'exp', 'stress-representation'))
from paradigm import ParadigmModel, stress_class, ACC, NC
from validate import load_splits, weighted_score, DATA
import eval_reps
from eval_reps import REPS, MaskParadigmModel
eval_reps.register_main()

SEGS = ['transfer', 'complete', 'wug', 'unseen']
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
BASE_CFG = {s: dict(rep='cur5', lam=2.0, joint=True) for s in SEGS}
BEST_CFG = {'transfer': dict(rep='cur5', lam=2.0, joint=True),
            'complete': dict(rep='3way', lam=5.0, joint=True),
            'wug': dict(rep='3way', lam=5.0, joint=True),
            'unseen': dict(rep='cur5', lam=2.0, joint=False)}

p = argparse.ArgumentParser()
p.add_argument('--mode', default='dev', choices=['dev', 'full'])
p.add_argument('--ext', default=None)
p.add_argument('--out', default=None)
p.add_argument('--cfg', default='best', help="'best' | 'base' | JSON dict seg->{rep,lam,joint}")
p.add_argument('--mu', type=float, default=0.6)
p.add_argument('--K5', type=int, default=600)
p.add_argument('--s5', type=int, default=6)
p.add_argument('--K3', type=int, default=300)
p.add_argument('--s3', type=int, default=3)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--grid', action='store_true', help='dev: also sweep lam/mu per segment and rep')
p.add_argument('--ctx_clean', type=int, default=0, help='drop flagged substitution-noise rows from the paradigm context')
p.add_argument('--stem_cons', default='', help='comma list of segments where stem-prefix consensus is applied')
p.add_argument('--cons_cut', type=int, default=2, help='chars dropped from the canonical stem end')
p.add_argument('--models', default='', help='comma list of local class-conditioned model dirs to pool')
p.add_argument('--w_loc', default='0.5', help='comma list of local pooling weights (dev: all evaluated)')
p.add_argument('--beam', type=int, default=4)
p.add_argument('--device', default='mps')
p.add_argument('--fixes', default='', help="post-fixes to apply in order, subset of 'BAC'")
p.add_argument('--dump', default='', help='dev: pickle per-row predictions/golds/pools after fixes')
p.add_argument('--clf', default='', help='comma list of stress-classifier run dirs (wug expert)')
p.add_argument('--rescored', default='', help='comma list of rescoring pkls (key -> {form: logp}); replaces beam tallies')
p.add_argument('--resc_mix', type=float, default=0.0, help='weight of the original beam tally mixed into the rescored tally')
p.add_argument('--resc_temp', type=float, default=1.0, help='temperature on rescored log-probs')
p.add_argument('--clf_w', default='0', help='comma list of weights for the classifier expert (dev: all evaluated)')
p.add_argument('--leak', type=int, default=0, help='1: use the publicly disclosed noise-copy leak (forum topic 743193): overrides + prior observations')
p.add_argument('--leak_ctx', type=int, default=1, help='with --leak: also add the matched forms to the paradigm-prior observations')
p.add_argument('--leak_lem', default='', help='thr,margin: tie copies not matched by pools to a lemma by spelling (ctx; override if the cell is a row)')
p.add_argument('--cache_tag', default='', help='suffix for the local-beam cache file (parallel runs)')
p.add_argument('--obs_cfg', default='', help='JSON {rep,lam,joint,lam5}: decision config for unseen lemmas that received leaked observations')
p.add_argument('--leak_stem', type=int, default=0, help='with --leak: also give matched forms to the local models as context (re-decodes those lemmas)')
args = p.parse_args()
cfg = BEST_CFG if args.cfg == 'best' else BASE_CFG if args.cfg == 'base' else json.loads(args.cfg)
t0 = time.time()

tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv')
wug['lemma_frequency'] = 60.0
if args.mode == 'dev':
    train_rows, known, val = load_splits()
    pm_fit = train_rows
    ext = pickle.load(open(args.ext or os.path.join(ROOT, 'runs/kdev2/val_cands_aligned.pkl'), 'rb'))
    seg_frames = {s: (val[s], ext[KEY[s]]) for s in SEGS}
else:
    known = pd.concat([tr, dv, wug]); pm_fit = pd.concat([tr, dv])
    ext = pickle.load(open(args.ext or os.path.join(ROOT, 'runs/test_cands_v1v3.pkl'), 'rb'))
    lf = set(zip(pm_fit.lemma_ru, pm_fit.feats)); trl = set(pm_fit.lemma_ru); wl = set(wug.lemma_ru)
    seg_of = np.array(['transfer' if (l, f) in lf else 'complete' if l in trl else 'wug' if l in wl else 'unseen'
                       for l, f in zip(te.lemma_ru, te.feats)])
    seg_frames = {}
    for s in SEGS:
        idx = np.where(seg_of == s)[0]
        seg_frames[s] = (te.iloc[idx], [[m[i] for i in idx] for m in ext])
if args.limit:
    seg_frames = {s: (d.head(args.limit), [m[:args.limit] for m in c]) for s, (d, c) in seg_frames.items()}

CLF = {}
if args.clf:
    _src_df = val['wug'] if args.mode == 'dev' else te
    for cd in args.clf.split(','):
        fn = os.path.join(ROOT, cd, 'holdout_wug_probs.pkl' if args.mode == 'dev' else 'test_wug_probs.pkl')
        dd = pickle.load(open(fn, 'rb'))
        for ix, pr in zip(dd['index'], dd['P']):
            r = _src_df.loc[ix]
            k = (r.lemma_ru, r.feats, r.dialect)
            CLF.setdefault(k, []).append(np.asarray(pr))
    CLF = {k: np.exp(np.mean([np.log(p_ + 1e-6) for p_ in v], axis=0)) for k, v in CLF.items()}
    print(f'classifier expert: {len(CLF)} rows from {args.clf}', flush=True)
known_ctx = known
if args.ctx_clean:
    sys.path.insert(0, os.path.join(ROOT, 'src', 'exp', 'noise'))
    from clean_known import clean_known
    known_ctx = clean_known(known)
    print(f'ctx_clean: {len(known)} -> {len(known_ctx)} context rows', flush=True)
known_map = collections.defaultdict(list)
for l, f, d, fm in zip(known_ctx.lemma_ru, known_ctx.feats, known_ctx.dialect, known_ctx.form_vvz):
    known_map[l].append((f, d, fm))
# full (uncleaned) forms for stem references
stem_ref_map = collections.defaultdict(list)
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    stem_ref_map[l].append((f, d, fm))

# ---------------------------------------------------------------- noise-copy leak (disclosed publicly: forum topic 743193)
# Substitution-noise rows in train/dev carry the exact form (stress included) of another row with the same feats and
# dialect. Copies that match no released row are matched to the unique unreleased row (same feats, dialect) whose
# candidate pool contains the same unaccented string.
LEAK = {}
LEAK_STEM_LEMMAS = set()
if args.leak:
    _fl = pd.read_csv(os.path.join(ROOT, 'runs/exp/noise/sub_flags_traindev.csv'))
    _bad = set(zip(_fl.lemma_ru[_fl.sub2], _fl.feats[_fl.sub2], _fl.dialect[_fl.sub2], _fl.form_vvz[_fl.sub2]))
    _ksub = np.array([(l, f, d, fm) in _bad for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz)])
    _legit = set(known.form_vvz.values[~_ksub])
    _U = known[_ksub & ~known.form_vvz.isin(_legit).values]
    _pfn = os.path.join(ROOT, f'runs/final/pool_{args.mode}.pkl')
    _xpool = pickle.load(open(_pfn, 'rb')) if os.path.exists(_pfn) else {}
    _idx = collections.defaultdict(set)
    for _s in SEGS:
        _df, _cpm = seg_frames[_s]
        for _i, _r in enumerate(_df.itertuples()):
            _key = (_r.lemma_ru, _r.feats, _r.dialect)
            _cs = {c[0] for m_ in _cpm for c in m_[_i]} | set(_xpool.get(_key, []))
            for c in _cs:
                if c: _idx[(_r.feats, _r.dialect, c.replace(ACC, ''))].add(_key)
    _cand = collections.defaultdict(set)
    for _r in _U.itertuples():
        _h = _idx.get((_r.feats, _r.dialect, _r.form_vvz.replace(ACC, '')), set())
        if len(_h) == 1: _cand[next(iter(_h))].add(_r.form_vvz)
    LEAK = {k: next(iter(v)) for k, v in _cand.items() if len(v) == 1}
    _nlem = collections.Counter()
    if args.leak_lem:
        sys.path.insert(0, os.path.join(ROOT, 'src', 'exp', 'noise'))
        from noise_detect import canon_form as _cf, canon_lemma as _cl
        from rapidfuzz import process as _proc, fuzz as _fz
        _thr, _mg = [float(x) for x in args.leak_lem.split(',')]
        _lem = pd.concat([x_[['lemma_ru', 'pos']] for x_ in (tr, dv, wug, te)]).drop_duplicates()
        _byp = {p_: (g.lemma_ru.tolist(), [_cl(x) for x in g.lemma_ru]) for p_, g in _lem.groupby('pos')}
        _rowkeys = set(k_ for v_ in _idx.values() for k_ in v_)
        _used = set(f_ for v_ in _cand.values() for f_ in v_)
        _cand2 = collections.defaultdict(set); _ctx2 = []
        for _r in _U.itertuples():
            if _r.form_vvz in _used: continue
            _nm, _cls = _byp[_r.feats.split(';')[0]]
            _res = _proc.extract(_cf(_r.form_vvz), _cls, scorer=_fz.ratio, limit=2)
            if len(_res) < 2 or _res[0][1] < _thr or _res[0][1] - _res[1][1] < _mg: continue
            _L = _nm[_res[0][2]]
            if _L == _r.lemma_ru: continue
            _k = (_L, _r.feats, _r.dialect)
            if _k in _rowkeys: _cand2[_k].add(_r.form_vvz)
            else: _ctx2.append((_L, _r.feats, _r.dialect, _r.form_vvz))
        for _k, _v in _cand2.items():
            if len(_v) == 1 and _k not in LEAK: LEAK[_k] = next(iter(_v)); _nlem['row'] += 1
        for _L, _f, _d, _fm in _ctx2: known_map[_L].append((_f, _d, _fm)); _nlem['ctx_only'] += 1
    if args.leak_ctx:
        for (l, f, d), fm in LEAK.items(): known_map[l].append((f, d, fm))
    if args.leak_stem:
        for (l, f, d), fm in LEAK.items(): stem_ref_map[l].append((f, d, fm)); LEAK_STEM_LEMMAS.add(l)
    print(f'leak: {len(_U)} unexplained noise copies -> {len(LEAK)} rows matched (ctx={args.leak_ctx}) lemma-matched {dict(_nlem)}', flush=True)

_CANON = str.maketrans({'о': 'а', 'е': 'е', 'и': 'е', 'ѣ': 'е', 'э': 'е', 'ё': 'е', 'ы': 'е', 'ь': None, 'ъ': None, ACC: None})


def canon(form):
    return form.translate(_CANON)


def lemma_stem_len(lemma, pos):
    l = lemma
    if pos == 'ADJ':
        if l.endswith('ся'): l = l[:-2]
        l = l[:-2] if l[-2:] in ('ый', 'ий', 'ой') else l
    elif pos == 'N':
        l = l[:-1] if l[-1] in 'аяоеёьй' else l
    else:
        for suf in ('ться', 'тись', 'чься', 'ть', 'ти', 'чь'):
            if l.endswith(suf): l = l[:-len(suf)]; break
    return len(canon(l))

# length prior P(len(form) - len(lemma) | pos, feats, dialect), backoff (pos, feats)
lencnt = collections.defaultdict(collections.Counter)
for l, p_, f, d, fm in zip(known.lemma_ru, known.pos, known.feats, known.dialect, known.form_vvz):
    lencnt[(p_, f, d)][len(fm.replace(ACC, '')) - len(l)] += 1
lencnt2 = collections.defaultdict(collections.Counter)
for (p_, f, d), c in list(lencnt.items()): lencnt2[(p_, f)].update(c)


def loglen(pos, feats, dialect, lemma, form):
    c = lencnt.get((pos, feats, dialect)) or lencnt2.get((pos, feats)) or collections.Counter()
    tot = sum(c.values())
    return np.log((c.get(len(form.replace(ACC, '')) - len(lemma), 0) + 0.5) / (tot + 0.5 * 31))


# ---------------------------------------------------------------- paradigm priors
def load_or_fit(kind, K, seeds):
    tag = 'dev' if args.mode == 'dev' else 'full'
    out = []
    for s in seeds:
        if args.mode == 'dev' and kind == 'cur5':
            fn = os.path.join(ROOT, f'runs/exp/decision-tuning/pm_K{K}_s{s}.pkl')
        elif args.mode == 'dev' and kind == '3way':
            fn = os.path.join(ROOT, f'runs/exp/stress-representation/pm_3way_K{K}_s{s}.pkl')
        else:
            fn = os.path.join(ROOT, f'runs/final/pm_{kind}_K{K}_s{s}_{tag}.pkl')
        if os.path.exists(fn):
            out.append(pickle.load(open(fn, 'rb'))); continue
        t = time.time()
        pm = ParadigmModel(K=K, seed=s).fit(pm_fit) if kind == 'cur5' else MaskParadigmModel(REPS['3way'], K=K, seed=s).fit(pm_fit)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        if args.mode == 'full': pickle.dump(pm, open(fn, 'wb'))
        print(f'  fit {kind} K={K} seed={s} in {time.time() - t:.0f}s', flush=True)
        out.append(pm)
    return out


need = {r for c in cfg.values() for r in (('cur5', '3way') if c['rep'] == 'poe' else (c['rep'],))} | ({'cur5', '3way'} if args.grid else set())
PMS = {}
if 'cur5' in need: PMS['cur5'] = load_or_fit('cur5', args.K5, range(args.s5))
if '3way' in need: PMS['3way'] = load_or_fit('3way', args.K3, range(args.s3))
print(f'priors ready {time.time() - t0:.0f}s', flush=True)


def cls_mask(rep, form, lemma, pos, dialect):
    if rep == 'cur5':
        c = stress_class(form)
        return (1 << c) if c >= 0 else 0
    return REPS[rep].mask(form, lemma, pos, dialect)


def ncls(rep): return NC if rep == 'cur5' else REPS[rep].n


def prior(rep, pos, feats, lemma, excl_dialects):
    obs = [(f, d, fm) for f, d, fm in known_map.get(lemma, []) if not (f == feats and d in excl_dialects)]
    if rep == 'cur5':
        return sum(pm.class_dist(pos, feats, [(f, fm) for f, d, fm in obs]) for pm in PMS['cur5']) / len(PMS['cur5'])
    return sum(pm.class_dist(pos, feats, [(f, REPS[rep].mask(fm, lemma, pos, d)) for f, d, fm in obs]) for pm in PMS[rep]) / len(PMS[rep])


RESC = [pickle.load(open(os.path.join(ROOT, f_), 'rb')) for f_ in args.rescored.split(',') if f_]
if RESC: print(f'rescored tallies from {len(RESC)} models', flush=True)


def prep(df, cpm):
    rows = list(df.itertuples())
    tallies = []
    for i in range(len(rows)):
        t = collections.defaultdict(float)
        for mdl in cpm:
            lst = mdl[i]; lp = np.array([c[1] for c in lst]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
            for (s, _), q in zip(lst, pr): t[s] += q / len(cpm)
        if RESC:
            key = (rows[i].lemma_ru, rows[i].feats, rows[i].dialect)
            new = collections.defaultdict(float); nm = 0
            for sc in RESC:
                d = sc.get(key)
                if not d: continue
                ks = list(d); lp = np.array([d[k_] for k_ in ks]) / args.resc_temp
                pr = np.exp(lp - lp.max()); pr /= pr.sum(); nm += 1
                for k_, q in zip(ks, pr): new[k_] += q
            if nm:
                new = {k_: v / nm for k_, v in new.items()}
                if args.resc_mix > 0:
                    for k_ in set(new) | set(t): new[k_] = (1 - args.resc_mix) * new.get(k_, 0.0) + args.resc_mix * t.get(k_, 0.0)
                t = new
        tallies.append(dict(t))
    ll = [{s: loglen(r.pos, r.feats, r.dialect, r.lemma_ru, s) for s in t} for r, t in zip(rows, tallies)]
    return rows, tallies, ll


_prior_cache = {}


def decide_poe(rows, tallies, ll, lam, joint, mu, lam5):
    """Product of experts: decide the 3way class per group with candidate mass reweighted by the cur5 prior^lam5."""
    n = ncls('3way')
    masks = [{s: cls_mask('3way', s, r.lemma_ru, r.pos, r.dialect) for s in t} for r, t in zip(rows, tallies)]
    c5 = [{s: stress_class(s) for s in t} for t in tallies]
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats) if joint else i].append(i)
    out = [None] * len(rows); chosen = [0] * len(rows)
    for g, idx in groups.items():
        r0 = rows[idx[0]]
        dls = tuple(sorted({rows[i].dialect for i in idx}))
        k3 = ('3way', r0.lemma_ru, r0.feats, dls); k5 = ('cur5', r0.lemma_ru, r0.feats, dls)
        if k3 not in _prior_cache: _prior_cache[k3] = prior('3way', r0.pos, r0.feats, r0.lemma_ru, set(dls))
        if k5 not in _prior_cache: _prior_cache[k5] = prior('cur5', r0.pos, r0.feats, r0.lemma_ru, set(dls))
        p3, p5 = _prior_cache[k3], _prior_cache[k5]
        sc = lam * np.log(p3 + 1e-9)
        w = [{s: q * (p5[c5[i][s]] if c5[i][s] >= 0 else 1e-3) ** lam5 for s, q in tallies[i].items()} for i in idx]
        for k, i in enumerate(idx):
            m = np.full(n, 1e-12)
            for s, v in w[k].items():
                for c in range(n):
                    if (masks[i][s] >> c) & 1: m[c] += v
            sc = sc + np.log(m / m.sum())
        c = int(sc.argmax())
        for k, i in enumerate(idx):
            cands = [(s, np.log(v + 1e-12) + mu * ll[i][s]) for s, v in w[k].items() if (masks[i][s] >> c) & 1]
            if not cands: cands = [(s, np.log(v + 1e-12) + mu * ll[i][s]) for s, v in w[k].items()]
            out[i] = max(cands, key=lambda x: x[1])[0]; chosen[i] = c
    decide.last = (masks, chosen)
    return out


def decide(rows, tallies, ll, rep, lam, joint, mu, cons=False, lam5=0.0, clfw=0.0):
    if rep == 'poe':
        return decide_poe(rows, tallies, ll, lam, joint, mu, lam5)
    n = ncls(rep)
    chosen_cls = [0] * len(rows)
    masks = [{s: cls_mask(rep, s, r.lemma_ru, r.pos, r.dialect) for s in t} for r, t in zip(rows, tallies)]
    mass = []
    for mk, t in zip(masks, tallies):
        m = np.full(n, 1e-9)
        for s, q in t.items():
            for c in range(n):
                if (mk[s] >> c) & 1: m[c] += q
        mass.append(m)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats) if joint else i].append(i)
    out = [None] * len(rows)
    for g, idx in groups.items():
        r0 = rows[idx[0]]
        dls = tuple(sorted({rows[i].dialect for i in idx}))
        key = (rep, r0.lemma_ru, r0.feats, dls)
        if key not in _prior_cache: _prior_cache[key] = prior(rep, r0.pos, r0.feats, r0.lemma_ru, set(dls))
        pr = _prior_cache[key]
        sc = sum(np.log(mass[i]) for i in idx) + lam * np.log(pr + 1e-9)
        if clfw and rep == '3way':
            for i in idx:
                cp = CLF.get((rows[i].lemma_ru, rows[i].feats, rows[i].dialect))
                if cp is not None: sc = sc + clfw * np.log(cp[:n] + 1e-6)
        c = int(sc.argmax())
        for i in idx:
            cands = [(s, np.log(q + 1e-12) + mu * ll[i][s]) for s, q in tallies[i].items() if (masks[i][s] >> c) & 1]
            if not cands: cands = [(s, np.log(q + 1e-12) + mu * ll[i][s]) for s, q in tallies[i].items()]
            out[i] = max(cands, key=lambda x: x[1])[0]
            chosen_cls[i] = c
    if cons:
        out = stem_consensus(rows, tallies, ll, masks, chosen_cls, out, mu)
    decide.last = (masks, chosen_cls)
    return out


OBS_CFG = json.loads(args.obs_cfg) if args.obs_cfg else None
OBS_LEMMAS = set()
if OBS_CFG and LEAK:
    _uns = set(seg_frames['unseen'][0].lemma_ru)
    OBS_LEMMAS = {l_ for l_ in _uns if any(True for _ in known_map.get(l_, []))}
    print(f'obs_cfg: {len(OBS_LEMMAS)} unseen lemmas with leaked observations use {OBS_CFG}', flush=True)
_decide_base = decide


def decide(rows, tallies, ll, rep, lam, joint, mu, cons=False, lam5=0.0, clfw=0.0):
    A = [i for i, r in enumerate(rows) if r.lemma_ru in OBS_LEMMAS] if OBS_LEMMAS else []
    if not A:
        return _decide_base(rows, tallies, ll, rep, lam, joint, mu, cons, lam5, clfw)
    sA = set(A); B = [i for i in range(len(rows)) if i not in sA]
    n = len(rows); out = [None] * n; masks = [None] * n; chosen = [0] * n; reps = [rep] * n
    for idx, (rp, lm, jt, l5) in ((B, (rep, lam, joint, lam5)), (A, (OBS_CFG['rep'], OBS_CFG['lam'], OBS_CFG['joint'], OBS_CFG.get('lam5', 0.0)))):
        if not idx: continue
        o = _decide_base([rows[i] for i in idx], [tallies[i] for i in idx], [ll[i] for i in idx], rp, lm, jt, mu, cons, l5, clfw)
        mk, ch = decide.last[0], decide.last[1]
        for k, i in enumerate(idx): out[i] = o[k]; masks[i] = mk[k]; chosen[i] = ch[k]; reps[i] = rp
    decide.last = (masks, chosen, reps)
    return out


def stem_consensus(rows, tallies, ll, masks, cls_, out, mu):
    """Per lemma (per dialect for SEV pleophony lemmas): reference canonical stem prefix from the lemma's
    known forms if any, else the mass-weighted consensus of its rows; restrict each row to class-compatible
    candidates carrying that prefix when such candidates exist."""
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        pleo = any(x in r.lemma_ru for x in ('оро', 'ере', 'оло'))
        groups[(r.lemma_ru, r.dialect if pleo else '*')].append(i)
    new = list(out)
    for (lemma, dgrp), idx in groups.items():
        r0 = rows[idx[0]]
        P = max(2, lemma_stem_len(lemma, r0.pos) - args.cons_cut)
        refs = collections.Counter()
        for f, d, fm in stem_ref_map.get(lemma, []):
            if dgrp != '*' and d != dgrp: continue
            refs[canon(fm)[:P]] += 1
        if refs:
            ref = refs.most_common(1)[0][0]
        else:
            if len(idx) < 3: continue
            score = collections.defaultdict(float)
            for i in idx:
                best = {}
                for s_, q in tallies[i].items():
                    if (masks[i][s_] >> cls_[i]) & 1:
                        k = canon(s_)[:P]; best[k] = max(best.get(k, 0.0), q)
                for k, v in best.items(): score[k] += v
            if not score: continue
            ref = max(score.items(), key=lambda x: x[1])[0]
        for i in idx:
            if canon(out[i])[:P] == ref: continue
            cands = [(s_, np.log(q + 1e-12) + mu * ll[i][s_]) for s_, q in tallies[i].items()
                     if (masks[i][s_] >> cls_[i]) & 1 and canon(s_)[:P] == ref]
            if cands: new[i] = max(cands, key=lambda x: x[1])[0]
    return new



# ---------------------------------------------------------------- residual-error post-fixes (src/exp/residual-errors)
import re as _re
from rapidfuzz.distance import Levenshtein as _Lev
_VOW = 'аеиоуыэюяёѣѫ'
def _strip(x): return x.replace(ACC, '')
def _akn(x): return _strip(x).replace('о', 'а').replace('ѣ', 'е').replace('и', 'е').replace('ы', 'е')
def _cons(x):
    x = _strip(x).replace('-', ''); x = _re.sub('[' + _VOW + 'ьъй]', '', x)
    return x.translate(str.maketrans('цчзжсш', 'ккггхх'))
def _lemma_stem(lemma, pos):
    ends = {'ADJ': ('ый', 'ий', 'ой'), 'V': ('ться', 'ти', 'ть', 'чь'), 'N': ('ия', 'ие', 'а', 'я', 'о', 'е', 'ь', 'й')}[pos]
    for e in ends:
        if lemma.endswith(e): return lemma[:-len(e)]
    return lemma
_CONSL = 'бвгджзклмнпрстфхцчшщ'; _PAL = {'к': 'цч', 'г': 'зж', 'х': 'сш', 'ц': 'ч'}
def _gem_fix(lemma, pred):
    out = pred
    for m in _re.finditer(r'([' + _CONSL + r'])\1', lemma):
        i, c = m.start(), m.group(1)
        so = _strip(out)
        if c + c in so or c + 'ъ' + c in so or c + 'ь' + c in so: continue
        if any(c + sep + alt in so for sep in ('', 'ь') for alt in _PAL.get(c, '')): continue
        ctx = lemma[max(0, i - 3):i]
        pat = ''.join(('[оа]' if ch == 'о' else '[еѣи]' if ch == 'е' else _re.escape(ch)) + ACC + '?' for ch in ctx)
        mm = _re.search(pat + c + ACC + '?(?!' + c + ')', out)
        if mm:
            e = mm.end()
            out = out[:e - 1] + c + out[e - 1:] if out[e - 1] == ACC else out[:e] + c + out[e:]
    return out

def apply_fixes(segrows, fixes, w_loc):
    """segrows: dict seg -> list of dicts(row, pred, pool). pool = {form: score} of class-compatible candidates."""
    allrows = [x for s in SEGS for x in segrows.get(s, [])]
    kshort = {(l, f, d): fm for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz)}
    stats = collections.Counter()
    for fx in fixes:
        if fx == 'B':
            for x in allrows:
                r = x['row']; ls = _cons(_lemma_stem(r.lemma_ru, r.pos))
                if _cons(x['pred']).startswith(ls): continue
                c = [(s_, q) for s_, q in x['pool'].items() if _cons(s_).startswith(ls)]
                if c: x['pred'] = max(c, key=lambda z: z[1])[0]; stats['B'] += 1
        elif fx == 'A':
            pshort = {(x['row'].lemma_ru, x['row'].feats, x['row'].dialect): x['pred'] for x in allrows
                      if x['row'].pos == 'ADJ' and 'SHORT' in x['row'].feats}
            for x in allrows:
                r = x['row']
                if r.pos != 'ADJ' or 'LONG' not in r.feats: continue
                key = (r.lemma_ru, r.feats.replace('LONG', 'SHORT'), r.dialect)
                src = _strip(kshort[key]) if key in kshort else (_strip(pshort[key]) if key in pshort else None)
                if src is None: continue
                target = src + ('ии' if ';PL;' in r.feats else 'ыи')
                c = [(s_, q) for s_, q in x['pool'].items() if _strip(s_) == target]
                if not c: continue
                best = max(c, key=lambda z: z[1])[0]
                if best == x['pred']: continue
                ins = [o for o in _Lev.opcodes(_strip(x['pred']), target) if o.tag != 'equal']
                if len(ins) == 1 and ins[0].tag == 'insert' and _Lev.distance(_akn(x['pred']), _akn(target)) == 1:
                    x['pred'] = best; stats['A'] += 1
        elif fx == 'C':
            for x in allrows:
                new = _gem_fix(x['row'].lemma_ru, x['pred'])
                if new != x['pred']: x['pred'] = new; stats['C'] += 1
    return dict(stats)


def build_segrows(seg, rows, tallies, preds, rep, w_loc):
    """pool of class-compatible candidates per row (Kaggle mass + local beams if pooling ran)."""
    masks, cls_ = decide.last[0], decide.last[1]
    reps_ = decide.last[2] if len(decide.last) > 2 else None
    lb = getattr(pooled, 'lb', None) if LOCAL else None
    out = []
    for i, r in enumerate(rows):
        c = cls_[i]; rr_ = reps_[i] if reps_ else rep; mrep = '3way' if rr_ == 'poe' else rr_
        pool = collections.defaultdict(float)
        for s_, q in tallies[i].items():
            if (masks[i][s_] >> c) & 1: pool[s_] += q
        if lb is not None:
            for s_, q in lb[i].items():
                if (cls_mask(mrep, s_, r.lemma_ru, r.pos, r.dialect) >> c) & 1: pool[s_] += w_loc * q
        out.append(dict(row=r, pred=preds[i], pool=dict(pool)))
    return out

prepped = {s: prep(*seg_frames[s]) for s in SEGS}

# ---------------------------------------------------------------- local class-conditioned models (optional pooling)
LOCAL = []
if args.models:
    import torch
    from model import Seq2Seq
    _src = open(os.path.join(ROOT, 'src', 'local_train.py')).read()
    _ns = {'math': __import__('math'), 'random': __import__('random'), 'np': np, 'NC': NC}
    exec(_src[_src.index("ACC = "):_src.index("# ---------------- splits")], _ns)
    import __main__; __main__.Vocab = _ns['Vocab']   # meta.pkl pickled Vocab from local_train's __main__
    _dev = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
    for md in args.models.split(','):
        meta = pickle.load(open(os.path.join(ROOT, md, 'meta.pkl'), 'rb')); ta = meta['args']
        m = Seq2Seq(len(meta['vocab']), ta['d_model'], 4, ta['layers'], ta['layers'], ta['ff'], 0.0, max_len=512).to(_dev)
        m.load_state_dict(torch.load(os.path.join(ROOT, md, 'model_last.pt'), map_location=_dev)); m.eval()
        LOCAL.append((m, meta['vocab'], ta))
    _ns['args'] = argparse.Namespace(max_ctx=LOCAL[0][2]['max_ctx'])
    make_input = _ns['make_input']; BOS, EOS = _ns['BOS'], _ns['EOS']
    print(f'loaded {len(LOCAL)} local models on {_dev}', flush=True)


_LB_CACHE_FN = os.path.join(ROOT, 'runs', 'final', f'lb_cache_{args.mode}_' + '_'.join(os.path.basename(m) for m in args.models.split(',')) + f'_b{args.beam}{args.cache_tag}.pkl') if args.models else None
_LB_FRESH = set()
_LB_CACHE = pickle.load(open(_LB_CACHE_FN, 'rb')) if (_LB_CACHE_FN and os.path.exists(_LB_CACHE_FN)) else {}


def local_beams(rows, tokens):
    """Cached wrapper: key = (lemma, feats, dialect, token)."""
    keys = [(r.lemma_ru, r.feats, r.dialect, t) for r, t in zip(rows, tokens)]
    need = [i for i, (k, t) in enumerate(zip(keys, tokens)) if t is not None and (k not in _LB_CACHE or (k[0] in LEAK_STEM_LEMMAS and k not in _LB_FRESH))]
    if need:
        sub = _local_beams([rows[i] for i in need], [tokens[i] for i in need])
        for i, d in zip(need, sub): _LB_CACHE[keys[i]] = dict(d); _LB_FRESH.add(keys[i])
        pickle.dump(_LB_CACHE, open(_LB_CACHE_FN, 'wb'))
    print(f'    local beam cache: {len(need)} decoded, {len(rows) - len(need)} reused', flush=True)
    return [dict(_LB_CACHE.get(k, {})) if t is not None else {} for k, t in zip(keys, tokens)]


def _local_beams(rows, tokens):
    """rows: list of namedtuples; tokens: cur5 class token per row (None = skip). Returns list of {form: prob}."""
    import torch
    out = [collections.defaultdict(float) for _ in rows]
    todo = [i for i, t in enumerate(tokens) if t is not None]
    for m, vocab, ta in LOCAL:
        srcs = []
        for i in todo:
            r = rows[i]
            ctx = [c for c in stem_ref_map.get(r.lemma_ru, []) if not (c[0] == r.feats and c[1] == r.dialect)]
            srcs.append(vocab.enc(make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, tokens[i], ctx)))
        order = np.argsort([len(x) for x in srcs])
        with torch.no_grad():
            for b in range(0, len(order), 128):
                idx = order[b:b + 128]
                S = max(len(srcs[j]) for j in idx); x = torch.zeros(len(idx), S, dtype=torch.long)
                for k, j in enumerate(idx): x[k, :len(srcs[j])] = torch.tensor(srcs[j])
                ys, sc = m.beam(x.to(_dev), vocab.stoi[BOS], vocab.stoi[EOS], beam=args.beam, return_all=True)
                ys = ys.cpu().tolist(); sc = sc.cpu().numpy()
                for j, yy, cc in zip(idx, ys, sc):
                    pr = np.exp(cc - cc.max()); pr /= pr.sum()
                    for y, q in zip(yy, pr):
                        out[todo[j]][vocab.dec(y)] += float(q) / len(LOCAL)
    return out


def pooled(seg, rows, tallies, ll, rep, preds, mu, wlist):
    """Re-select each row's form from Kaggle candidates + local beams, restricted to the chosen class."""
    masks, cls_ = decide.last[0], decide.last[1]
    reps_ = decide.last[2] if len(decide.last) > 2 else None
    tokens = []
    for i, r in enumerate(rows):
        c = cls_[i]
        comp = [(s_, q) for s_, q in tallies[i].items() if (masks[i][s_] >> c) & 1]
        best = max(comp, key=lambda x: x[1])[0] if comp else preds[i]
        t = stress_class(best)
        tokens.append(t if t >= 0 else None)
    t1 = time.time(); lb = local_beams(rows, tokens); pooled.lb = lb
    print(f'  [{seg}] local beams {len(rows)} rows x {len(LOCAL)} models in {time.time() - t1:.0f}s', flush=True)
    res = {}
    for w in wlist:
        new = []
        for i, r in enumerate(rows):
            c = cls_[i]
            pool = collections.defaultdict(float)
            for s_, q in tallies[i].items():
                if (masks[i][s_] >> c) & 1: pool[s_] += q
            for s_, q in lb[i].items():
                rr_ = reps_[i] if reps_ else rep
                if (cls_mask('3way' if rr_ == 'poe' else rr_, s_, r.lemma_ru, r.pos, r.dialect) >> c) & 1: pool[s_] += w * q
            if not pool: new.append(preds[i]); continue
            new.append(max(pool.items(), key=lambda x: np.log(x[1] + 1e-12) + mu * loglen(r.pos, r.feats, r.dialect, r.lemma_ru, x[0]))[0])
        res[w] = new
    return res

print(f'candidates ready {time.time() - t0:.0f}s', flush=True)

if args.mode == 'dev':
    golds = {s: [r.form_vvz for r in prepped[s][0]] for s in SEGS}

    def run(cfg_, mu):
        preds = {s: decide(*prepped[s], cfg_[s]['rep'], cfg_[s]['lam'], cfg_[s]['joint'], mu, cons=(s in CONS), lam5=cfg_[s].get('lam5', 0.0)) for s in SEGS}
        return weighted_score(preds, golds)

    CONS = set()
    base = run(BASE_CFG, 0.0)
    CONS = set(x for x in args.stem_cons.split(',') if x)
    print('BASELINE (cur5 lam2 joint K5=%d mu=0):' % args.K5, base)
    res = run(cfg, args.mu)
    print('CONFIG', json.dumps(cfg), 'mu', args.mu, '->', res)
    print('DELTA weighted score: %+.5f' % (res[0] - base[0]))
    if CLF:
        for cw in [float(x) for x in args.clf_w.split(',')]:
            preds = {s: decide(*prepped[s], cfg[s]['rep'], cfg[s]['lam'], cfg[s]['joint'], args.mu, lam5=cfg[s].get('lam5', 0.0),
                               clfw=(cw if s == 'wug' else 0.0)) for s in SEGS}
            r_ = weighted_score(preds, golds)
            print(f'CLF w={cw}: wug EM {r_[1]["wug"][0]:.4f}  score {r_[0]:.5f}', flush=True)
    if LOCAL:
        wl = [float(x) for x in args.w_loc.split(',')]
        pp = {w: {} for w in wl}
        for s in SEGS:
            rows_, tall_, ll_ = prepped[s]
            pr_ = decide(rows_, tall_, ll_, cfg[s]['rep'], cfg[s]['lam'], cfg[s]['joint'], args.mu, cons=(s in CONS), lam5=cfg[s].get('lam5', 0.0))
            r_ = pooled(s, rows_, tall_, ll_, cfg[s]['rep'], pr_, args.mu, wl)
            for w in wl: pp[w][s] = r_[w]
            print(f'   {s}: no-pool EM {np.mean([a == b for a, b in zip(pr_, golds[s])]):.4f} ' +
                  ' '.join(f'w{w}:{np.mean([a == b for a, b in zip(r_[w], golds[s])]):.4f}' for w in wl), flush=True)
        for w in wl: print(f'POOLED w_loc={w}', weighted_score(pp[w], golds))
    if args.fixes:
        w0 = float(args.w_loc.split(',')[0])
        segrows = {}
        for s in SEGS:
            rows_, tall_, ll_ = prepped[s]
            pr_ = decide(rows_, tall_, ll_, cfg[s]['rep'], cfg[s]['lam'], cfg[s]['joint'], args.mu, cons=(s in CONS), lam5=cfg[s].get('lam5', 0.0))
            if LOCAL: pr_ = pooled(s, rows_, tall_, ll_, cfg[s]['rep'], pr_, args.mu, [w0])[w0]
            segrows[s] = build_segrows(s, rows_, tall_, pr_, cfg[s]['rep'], w0)
        before = {s: [x['pred'] for x in segrows[s]] for s in SEGS}
        print('BEFORE FIXES', weighted_score(before, golds))
        st = apply_fixes(segrows, args.fixes, w0)
        after = {s: [x['pred'] for x in segrows[s]] for s in SEGS}
        print('AFTER FIXES', args.fixes, st, weighted_score(after, golds))
        if LEAK:
            nfx = collections.Counter(); nbr = collections.Counter(); nov = collections.Counter()
            for s_ in SEGS:
                for x, g in zip(segrows[s_], golds[s_]):
                    k = (x['row'].lemma_ru, x['row'].feats, x['row'].dialect)
                    if k in LEAK and LEAK[k] != x['pred']:
                        nov[s_] += 1; nfx[s_] += (LEAK[k] == g); nbr[s_] += (x['pred'] == g); x['pred'] = LEAK[k]
            after = {s_: [x['pred'] for x in segrows[s_]] for s_ in SEGS}
            print('AFTER LEAK OVERRIDE', weighted_score(after, golds))
            print('   leak override changed', dict(nov), 'fixed', dict(nfx), 'broke', dict(nbr))
        if args.dump:
            pickle.dump(dict(leak=set(LEAK), obs_lemmas=set(l_ for l_, _, _ in LEAK) | set(_L for _L, _, _, _ in (_ctx2 if args.leak and args.leak_lem else []))), open(args.dump + '.leak', 'wb'))
            pickle.dump({s: [dict(lemma=x['row'].lemma_ru, pos=x['row'].pos, feats=x['row'].feats, dialect=x['row'].dialect,
                                  pred=x['pred'], gold=g, pool=x['pool']) for x, g in zip(segrows[s], golds[s])] for s in SEGS},
                        open(args.dump, 'wb'))
        for s in SEGS:
            fx = sum((a != b) and (b == g) for a, b, g in zip(before[s], after[s], golds[s]))
            br = sum((a != b) and (a == g) for a, b, g in zip(before[s], after[s], golds[s]))
            print(f'   {s}: fixed {fx} broke {br}')
    if args.grid:
        for s in SEGS:
            rows_ = []
            for rep in ('cur5', '3way'):
                for lam in (1, 2, 3, 5, 8):
                    for joint in (True, False):
                        for mu in (0.0, 0.6):
                            pr_ = decide(*prepped[s], rep, lam, joint, mu)
                            em = np.mean([a == b for a, b in zip(pr_, golds[s])])
                            rows_.append((em, rep, lam, joint, mu))
            rows_.sort(reverse=True)
            print(f'--- {s}: top configs (EM, rep, lam, joint, mu)')
            for r_ in rows_[:6]: print('   ', r_)
else:
    pred = np.empty(len(te), dtype=object)
    w0 = float(args.w_loc.split(',')[0])
    segrows = {}
    for s in SEGS:
        df = seg_frames[s][0]
        pr_ = decide(*prepped[s], cfg[s]['rep'], cfg[s]['lam'], cfg[s]['joint'], args.mu, cons=(s in set(x for x in args.stem_cons.split(',') if x)), lam5=cfg[s].get('lam5', 0.0), clfw=(float(args.clf_w.split(',')[0]) if s == 'wug' else 0.0))
        if LOCAL:
            pr2 = pooled(s, *prepped[s], cfg[s]['rep'], pr_, args.mu, [w0])[w0]
            print(f'  {s}: pooling changed {sum(a != b for a, b in zip(pr_, pr2))} rows', flush=True)
            pr_ = pr2
        segrows[s] = build_segrows(s, prepped[s][0], prepped[s][1], pr_, cfg[s]['rep'], w0)
        print(f'  {s}: {len(pr_)} rows decided', flush=True)
    if args.fixes:
        before = {s: [x['pred'] for x in segrows[s]] for s in SEGS}
        st = apply_fixes(segrows, args.fixes, w0)
        print('fixes applied:', st, {s: sum(a != x['pred'] for a, x in zip(before[s], segrows[s])) for s in SEGS}, flush=True)
    if LEAK:
        nov = collections.Counter()
        for s in SEGS:
            for x in segrows[s]:
                k = (x['row'].lemma_ru, x['row'].feats, x['row'].dialect)
                if k in LEAK and LEAK[k] != x['pred']: x['pred'] = LEAK[k]; nov[s] += 1
        print('leak override changed', dict(nov), flush=True)
    for s in SEGS:
        pred[seg_frames[s][0].index.values] = [x['pred'] for x in segrows[s]]
    assert all(isinstance(x, str) and x for x in pred)
    sub = pd.DataFrame({'id': te.id, 'form_vvz': pred})
    sub.to_csv(args.out, index=False)
    print('wrote', args.out, len(sub), f'{time.time() - t0:.0f}s')
