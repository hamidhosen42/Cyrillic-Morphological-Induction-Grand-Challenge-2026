"""Ensemble beam candidates from Kaggle runs + explicit paradigm re-ranking.

Usage:
  python postprocess.py --cands runs/kfull/test_cands.pkl --mode full --out runs/sub_x.csv
  python postprocess.py --cands runs/kdev/val_cands.pkl --mode dev   (evaluates on validation splits)

Decision rule: rows sharing (lemma, feats) must share a stress class, so the class is chosen
jointly per group:  argmax_c  sum_rows log P_nn(class=c | row) + lam * log P_paradigm(c)
then each row takes its best NN candidate of that class.
"""
import argparse, pickle, random, collections, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from paradigm import ParadigmModel, stress_class, NC

p = argparse.ArgumentParser()
p.add_argument('--cands', required=True)
p.add_argument('--mode', default='full')
p.add_argument('--out', default=None)
p.add_argument('--lam', type=float, default=2.0, help='lambda for train-lemma rows')
p.add_argument('--lam_new', type=float, default=1.0, help='lambda for wug/unseen rows')
p.add_argument('--K', type=int, default=300)
p.add_argument('--pm_seeds', default='0,1,2')
p.add_argument('--no_joint', action='store_true')
args = p.parse_args()

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv')
wug['lemma_frequency'] = 60.0

rng = random.Random(123)
if args.mode == 'dev':
    lem_adj = sorted(set(tr[tr.pos == 'ADJ'].lemma_ru))
    seedrows = tr[((tr.feats == 'N;ACC;SG') | (tr.feats == 'V;PRS;3;SG')) & (tr.dialect == 'SEV')]
    hold_wug = set(rng.sample(sorted(set(seedrows.lemma_ru)), 500))
    hold_adj = set(rng.sample(lem_adj, 250))
    train_rows = tr[~tr.lemma_ru.isin(hold_wug | hold_adj)]
    known = train_rows
    lf = set(zip(known.lemma_ru, known.feats))
    seen = np.array([(l, f) in lf for l, f in zip(dv.lemma_ru, dv.feats)])
    w = tr[tr.lemma_ru.isin(hold_wug)]
    wseed = w[((w.feats == 'N;ACC;SG') | (w.feats == 'V;PRS;3;SG')) & (w.dialect == 'SEV')]
    val_sets = {'dev_transfer': dv[seen], 'dev_complete': dv[~seen], 'wug': w.drop(wseed.index), 'unseen_adj': tr[tr.lemma_ru.isin(hold_adj)]}
    known = pd.concat([known, wseed])
    pm_fit = train_rows
else:
    known = pd.concat([tr, dv, wug])
    val_sets = {}
    pm_fit = pd.concat([tr, dv])

known_map = {}
for l, f, dl, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, dl, fm))
train_lemmas = set(pm_fit.lemma_ru)
pms = [ParadigmModel(K=args.K, seed=s).fit(pm_fit) for s in map(int, args.pm_seeds.split(','))]


def class_prior(pos, feats, obs):
    d = sum(pm.class_dist(pos, feats, obs) for pm in pms) / len(pms)
    return d


def combine(cands_per_model, i):
    tally = collections.defaultdict(float)
    for m in cands_per_model:
        lst = m[i]
        lp = np.array([c[1] for c in lst]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
        for (s, _), q in zip(lst, pr):
            tally[s] += q / len(cands_per_model)
    return tally


def decide(df, cands_per_model, exclude_self, joint=True):
    rows = list(df.itertuples())
    tallies = [combine(cands_per_model, i) for i in range(len(rows))]
    # per-row class mass and prior
    cmass, prior = [], []
    for r, t in zip(rows, tallies):
        m = np.full(NC, 1e-9)
        for s, q in t.items():
            c = stress_class(s)
            if c >= 0: m[c] += q
        cmass.append(m)
        obs = [(f, fm) for f, d, fm in known_map.get(r.lemma_ru, []) if not (exclude_self and f == r.feats and d == r.dialect)]
        prior.append(class_prior(r.pos, r.feats, obs))
    # groups
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.lemma_ru, r.feats) if joint else i].append(i)
    out = [None] * len(rows)
    for g, idx in groups.items():
        r0 = rows[idx[0]]
        lam = args.lam if r0.lemma_ru in train_lemmas else args.lam_new
        score = sum(np.log(cmass[i]) for i in idx) + lam * np.log(prior[idx[0]] + 1e-9)
        c = int(score.argmax())
        for i in idx:
            best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
            if best is None:
                best = max(tallies[i].items(), key=lambda x: x[1])
            out[i] = best[0]
    return out


cands = pickle.load(open(args.cands, 'rb'))
if args.mode == 'dev':
    for k, v in val_sets.items():
        cpm = cands[k]; golds = list(v.form_vvz)
        base = [max(combine(cpm, i).items(), key=lambda x: x[1])[0] for i in range(len(golds))]
        em0 = np.mean([a == b for a, b in zip(base, golds)])
        rr = decide(v, cpm, True, joint=not args.no_joint)
        em1 = np.mean([a == b for a, b in zip(rr, golds)])
        rr2 = decide(v, cpm, True, joint=False)
        em2 = np.mean([a == b for a, b in zip(rr2, golds)])
        print(f'{k:14s} n={len(golds)} models={len(cpm)}: NN-only EM {em0:.4f} | rerank joint {em1:.4f} | rerank per-row {em2:.4f}')
else:
    rr = decide(te, cands, False, joint=not args.no_joint)
    base = [max(combine(cands, i).items(), key=lambda x: x[1])[0] for i in range(len(rr))]
    sub = pd.DataFrame({'id': te.id, 'form_vvz': rr})
    sub.to_csv(args.out, index=False)
    print('changed vs NN-only:', np.mean([a != b for a, b in zip(base, rr)]), 'wrote', args.out)
