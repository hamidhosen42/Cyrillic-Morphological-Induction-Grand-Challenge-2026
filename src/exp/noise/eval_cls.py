"""Prior-only class accuracy + base-pipeline EM per holdout segment for a ParadigmModel configuration.
Usage: python3 src/exp/noise/eval_cls.py --alpha 1.0 [--drop runs/exp/noise/train_noise.csv --drop_cols sub,stressx2] [--seeds 0,1,2]
"""
import argparse, os, sys, pickle, collections, time
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from paradigm import ParadigmModel, stress_class, NC
from paradigm2 import ParadigmModel2
from joint import combine_tally, build_masses
from validate import load_splits, weighted_score, DATA

p = argparse.ArgumentParser()
p.add_argument('--alpha', type=float, default=1.0)
p.add_argument('--prior_alpha', type=float, default=1.0)
p.add_argument('--K', type=int, default=300)
p.add_argument('--seeds', default='0,1,2')
p.add_argument('--drop', default='')            # noise csv aligned to data/train.csv
p.add_argument('--drop_cols', default='sub')    # comma list among sub, stressx2 (minority), stressx1 (tie), typo, nacc
p.add_argument('--lam', type=float, default=2.0)
p.add_argument('--lam_new', type=float, default=2.0)
p.add_argument('--ext', default='runs/kdev2/val_cands_aligned.pkl')
p.add_argument('--clean_gold', default='')      # noise csv for gold filtering (train rows) to report clean-only accuracy
p.add_argument('--tag', default='')
p.add_argument('--ctx_clean', type=int, default=1)   # 1: drop flagged rows from the observation context too
p.add_argument('--fit_clean', type=int, default=1)   # 1: drop flagged rows from the paradigm fit
args = p.parse_args()

train_rows, known, val = load_splits()
tr = pd.read_csv(f'{DATA}/train.csv')
fit_rows = train_rows
if args.drop:
    nz = pd.read_csv(args.drop)
    assert len(nz) == len(tr)
    mask = np.zeros(len(tr), dtype=bool)
    for c in args.drop_cols.split(','):
        if c == 'sub': mask |= nz['sub'].values
        elif c == 'stressx2': mask |= (nz.stressx.values == 2)
        elif c == 'stressx1': mask |= (nz.stressx.values == 1)
        elif c == 'typo': mask |= nz.typo.values
        elif c == 'nacc': mask |= (nz.nacc.values != 1)
        elif c == 'pmx': mask |= nz.pmx.values
        else: mask |= nz[c].values.astype(bool)
    keep_ids = set(tr.id[~mask])
    fit_rows = train_rows[train_rows.id.isin(keep_ids)] if args.fit_clean else train_rows
    print(f'dropping {mask.sum()} of {len(tr)} train rows ({args.drop_cols}); fit rows {len(fit_rows)} of {len(train_rows)}')

known_map = {}
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, d, fm))
# if dropping, also remove dropped rows from the observation context (known_map) -- but keep wug seeds
if args.drop and args.ctx_clean:
    drop_ids = set(tr.id[mask])
    km2 = {}
    kk = known[~known.id.isin(drop_ids) | known.index.isin(known.index[known.lemma_ru.isin(set(val['wug'].lemma_ru))])]
    for l, f, d, fm in zip(kk.lemma_ru, kk.feats, kk.dialect, kk.form_vvz):
        km2.setdefault(l, []).append((f, d, fm))
    known_map_ctx = km2
else:
    known_map_ctx = known_map

t0 = time.time()
seeds = [int(s) for s in args.seeds.split(',')]
if args.alpha == 1.0 and args.prior_alpha == 1.0:
    pms = [ParadigmModel(K=args.K, seed=s).fit(fit_rows) for s in seeds]
else:
    pms = [ParadigmModel2(K=args.K, seed=s, alpha=args.alpha, prior_alpha=args.prior_alpha).fit(fit_rows) for s in seeds]
print(f'fit {len(pms)} pms in {time.time() - t0:.0f}s')
train_lemmas = set(train_rows.lemma_ru)

gold_clean = None
if args.clean_gold:
    nz = pd.read_csv(args.clean_gold)
    gold_clean = {}
    for i, s, x, t in zip(tr.id, nz['sub'], nz.stressx, nz.typo):
        gold_clean[i] = not (s or x == 2 or t)

ext = pickle.load(open(args.ext, 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
preds, golds, res = {}, {}, {}
for seg in ['transfer', 'complete', 'wug', 'unseen']:
    df = val[seg]; rows = list(df.itertuples())
    cpm = ext[KEY[seg]]
    tallies = [combine_tally(cpm, i) for i in range(len(rows))]
    masses = build_masses(tallies)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    gcls = [stress_class(r.form_vvz) for r in rows]
    prior_ok, comb_ok, nn_ok = [], [], []
    out = [None] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        lam = args.lam if lemma in train_lemmas else args.lam_new
        obs = [(f, fm) for f, d, fm in known_map_ctx.get(lemma, []) if not (f == feats and any(d == rows[i].dialect for i in idx))]
        prior = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        sc = sum(np.log(masses[i]) for i in idx) + lam * np.log(prior + 1e-9)
        c = int(sc.argmax()); cp = int(prior.argmax())
        for i in idx:
            prior_ok.append(cp == gcls[i]); comb_ok.append(c == gcls[i]); nn_ok.append(int(masses[i].argmax()) == gcls[i])
            best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
            out[i] = best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]
    # reorder ok lists to row order
    order = [i for idx in groups.values() for i in idx]
    po = np.zeros(len(rows), bool); co = np.zeros(len(rows), bool); no = np.zeros(len(rows), bool)
    po[order] = prior_ok; co[order] = comb_ok; no[order] = nn_ok
    em = np.array([o == r.form_vvz for o, r in zip(out, rows)])
    preds[seg] = out; golds[seg] = [r.form_vvz for r in rows]
    line = f'{seg:9s} n={len(rows):5d} prior-cls {po.mean():.4f} nn-cls {no.mean():.4f} comb-cls {co.mean():.4f} EM {em.mean():.4f}'
    if gold_clean is not None and seg in ('wug', 'unseen'):
        cm = np.array([gold_clean.get(r.id, True) for r in rows])
        line += f' | clean-gold n={cm.sum()} prior-cls {po[cm].mean():.4f} comb-cls {co[cm].mean():.4f} EM {em[cm].mean():.4f}'
    print(line, flush=True)
    res[seg] = dict(prior=po.mean(), comb=co.mean(), em=em.mean())
sc = weighted_score(preds, golds)
print('WEIGHTED', sc)
pickle.dump(dict(preds=preds, golds=golds, res=res, args=vars(args)), open(f'runs/exp/noise/eval_{args.tag or "base"}.pkl', 'wb'))
