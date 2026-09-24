"""Validate joint per-lemma paradigm inference against the per-cell decision, on test-like conditions.

The holdout wug/unseen lemmas have every cell; the test has only ~3.8 (wug) / ~7.3 (unseen) rows per
lemma. So rows are subsampled per lemma to match the test's per-lemma row-count distribution before
the joint inference (the inference only sees the subsampled rows).
"""
import argparse, os, sys, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from paradigm import ParadigmModel, stress_class, NC
from joint import JointParadigm, combine_tally, build_masses
from validate import load_splits, weighted_score, DATA

p = argparse.ArgumentParser()
p.add_argument('--ext', default='runs/kdev2/val_cands_aligned.pkl')
p.add_argument('--lam', type=float, default=2.0)
p.add_argument('--lam_new', type=float, default=2.0)
p.add_argument('--temp', type=float, default=1.0)
p.add_argument('--row_w', type=float, default=1.0)
p.add_argument('--K', type=int, default=300)
p.add_argument('--reps', type=int, default=3)
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

train_rows, known, val = load_splits()
te = pd.read_csv(f'{DATA}/test_features.csv'); tr = pd.read_csv(f'{DATA}/train.csv'); wugs = pd.read_csv(f'{DATA}/wug_seeds.csv')
known_map = {}
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, d, fm))
pms = [ParadigmModel(K=args.K, seed=s).fit(train_rows) for s in (0, 1, 2)]
train_lemmas = set(train_rows.lemma_ru)
ext = pickle.load(open(args.ext, 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}

# test per-lemma row-count distribution, per segment
lf = set(zip(tr.lemma_ru, tr.feats))
te_seg = np.where(te.lemma_ru.isin(set(tr.lemma_ru)), 'train',
                  np.where(te.lemma_ru.isin(set(wugs.lemma_ru)), 'wug', 'unseen'))
cnt_dist = {}
for seg in ['wug', 'unseen']:
    c = te[te_seg == seg].groupby('lemma_ru').size().values
    cnt_dist[seg] = c


def per_cell_decision(df, cpm, lam):
    """current pipeline: joint over (lemma, feats) only"""
    rows = list(df.itertuples())
    tallies = [combine_tally(cpm, i) for i in range(len(rows))]
    masses = build_masses(tallies)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r.lemma_ru, r.feats)].append(i)
    out = [None] * len(rows)
    for (lemma, feats), idx in groups.items():
        r0 = rows[idx[0]]
        obs = [(f, fm) for f, d, fm in known_map.get(lemma, []) if not (f == feats and d == r0.dialect)]
        prior = sum(pm.class_dist(r0.pos, feats, obs) for pm in pms) / len(pms)
        sc = sum(np.log(masses[i]) for i in idx) + lam * np.log(prior + 1e-9)
        c = int(sc.argmax())
        for i in idx:
            best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
            out[i] = best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]
    return out


def joint_decision(df, cpm, lam, sub_idx=None):
    """joint over the lemma; sub_idx: positions visible to the inference (others still predicted)"""
    rows = list(df.itertuples())
    tallies = [combine_tally(cpm, i) for i in range(len(rows))]
    masses = build_masses(tallies)
    jp = JointParadigm(pms, temp=args.temp, row_w=args.row_w)
    by_lemma = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_lemma[r.lemma_ru].append(i)
    visible = set(range(len(rows))) if sub_idx is None else set(sub_idx)
    out = [None] * len(rows)
    for lemma, idx in by_lemma.items():
        pos = rows[idx[0]].pos
        obs = [(f, stress_class(fm)) for f, d, fm in known_map.get(lemma, [])
               if not any(f == rows[i].feats and d == rows[i].dialect for i in idx)]
        cells = collections.defaultdict(list)
        for i in idx:
            cells[rows[i].feats].append(i)
        cell_list = list(cells)
        ev = [(f, np.mean([masses[i] for i in cells[f] if i in visible] or [np.ones(NC) / NC], axis=0)) for f in cell_list]
        dists = jp.class_dists(pos, obs, ev)
        for f, d in zip(cell_list, dists):
            for i in cells[f]:
                sc = np.log(masses[i]) + lam * np.log(d + 1e-9)
                c = int(sc.argmax())
                best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
                out[i] = best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]
    return out


rng = np.random.default_rng(args.seed)
res = collections.defaultdict(list)
for seg in ['transfer', 'complete', 'wug', 'unseen']:
    df = val[seg]; cpm = ext[KEY[seg]]; golds = list(df.form_vvz)
    lam = args.lam if seg in ('transfer', 'complete') else args.lam_new
    pc = per_cell_decision(df, cpm, lam)
    res['per_cell'].append((seg, np.mean([a == b for a, b in zip(pc, golds)])))
    if seg in ('wug', 'unseen'):
        # subsample rows per lemma like the test, several times
        pos_of = collections.defaultdict(list)
        for i, l in enumerate(df.lemma_ru): pos_of[l].append(i)
        accs = []
        for rep in range(args.reps):
            vis = []
            for l, idx in pos_of.items():
                k = min(len(idx), int(rng.choice(cnt_dist[seg])))
                vis.extend(rng.choice(idx, size=k, replace=False).tolist())
            jd = joint_decision(df, cpm, lam, sub_idx=vis)
            v = set(vis)
            accs.append(np.mean([jd[i] == golds[i] for i in range(len(golds)) if i in v]))
            if rep == 0:
                res['per_cell_visible'].append((seg, np.mean([pc[i] == golds[i] for i in range(len(golds)) if i in v])))
        res['joint'].append((seg, float(np.mean(accs))))
    else:
        jd = joint_decision(df, cpm, lam)
        res['joint'].append((seg, np.mean([a == b for a, b in zip(jd, golds)])))
print('temp', args.temp, 'row_w', args.row_w, 'lam_new', args.lam_new)
for k in ['per_cell', 'per_cell_visible', 'joint']:
    if res[k]: print(f'  {k:18s}', {s: round(v, 4) for s, v in res[k]})
