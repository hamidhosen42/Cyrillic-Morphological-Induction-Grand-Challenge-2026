"""Class-accuracy diagnostics for the wug segment (the only segment with real headroom).

Compares, on the wug holdout under test-like conditions (subsampled rows per lemma):
  prior       : paradigm posterior from the seed only (current pipeline)
  nn          : NN candidate class mass argmax
  comb        : log(nn) + lam*log(prior)                    <- current pipeline
  joint       : paradigm posterior from seed + NN mass of all visible rows of the lemma
  comb_joint  : log(nn) + lam*log(joint)
  oracle_par  : posterior with the lemma's TRUE paradigm known (headroom)
"""
import argparse, os, sys, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from paradigm import ParadigmModel, stress_class, NC
from joint import combine_tally, build_masses
from validate import load_splits, DATA

p = argparse.ArgumentParser()
p.add_argument('--ext', default='runs/kdev2/val_cands_aligned.pkl')
p.add_argument('--seg', default='wug')
p.add_argument('--lam', type=float, default=2.0)
p.add_argument('--K', type=int, default=300)
p.add_argument('--reps', type=int, default=3)
args = p.parse_args()

train_rows, known, val = load_splits()
te = pd.read_csv(f'{DATA}/test_features.csv'); tr = pd.read_csv(f'{DATA}/train.csv'); wugs = pd.read_csv(f'{DATA}/wug_seeds.csv')
known_map = {}
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, d, fm))
pms = [ParadigmModel(K=args.K, seed=s).fit(train_rows) for s in (0, 1, 2)]
ext = pickle.load(open(args.ext, 'rb'))
KEY = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
seg = args.seg
df = val[seg]; cpm = ext[KEY[seg]]
rows = list(df.itertuples())
tallies = [combine_tally(cpm, i) for i in range(len(rows))]
masses = build_masses(tallies)
gold = [stress_class(r.form_vvz) for r in rows]

te_seg = np.where(te.lemma_ru.isin(set(tr.lemma_ru)), 'train', np.where(te.lemma_ru.isin(set(wugs.lemma_ru)), 'wug', 'unseen'))
cnt_dist = te[te_seg == (seg if seg in ("wug","unseen") else "train")].groupby("lemma_ru").size().values

by_lemma = collections.defaultdict(list)
for i, r in enumerate(rows):
    by_lemma[r.lemma_ru].append(i)


def post_prior(pm, pos, obs):
    m = pm.models[pos]
    logp = np.log(m['prior']).copy()
    seen = {}
    for f, c in obs:
        if f in m['cidx'] and c >= 0: seen.setdefault(f, []).append(c)
    for f, cs in seen.items():
        c = collections.Counter(cs).most_common(1)[0][0]
        logp += np.log(m['P'][:, m['cidx'][f], c])
    w = np.exp(logp - logp.max()); return w / w.sum()


def add_rows(pm, pos, w0, ev, row_w, temp):
    m = pm.models[pos]
    logp = np.log(w0 + 1e-300)
    for f, mass in ev:
        if f not in m['cidx']: continue
        mm = np.power(np.maximum(mass, 1e-9), temp); mm /= mm.sum()
        logp += row_w * np.log((m['P'][:, m['cidx'][f], :] * mm[None, :]).sum(1) + 1e-12)
    w = np.exp(logp - logp.max()); return w / w.sum()


rng = np.random.default_rng(0)
acc = collections.defaultdict(list)
for rep in range(args.reps):
    for lemma, idx in by_lemma.items():
        k = min(len(idx), int(rng.choice(cnt_dist)))
        vis = rng.choice(idx, size=k, replace=False).tolist()
        pos = rows[idx[0]].pos
        obs = [(f, stress_class(fm)) for f, d, fm in known_map.get(lemma, [])]
        # true paradigm posterior (oracle): observe all gold classes of this lemma
        gobs = obs + [(rows[i].feats, gold[i]) for i in idx]
        cells = collections.defaultdict(list)
        for i in vis: cells[rows[i].feats].append(i)
        ev = [(f, np.mean([masses[i] for i in cells[f]], axis=0)) for f in cells]
        for pm in pms[:1]:
            m = pm.models[pos]
            w_pr = post_prior(pm, pos, obs)
            w_or = post_prior(pm, pos, gobs)
            for rw, tp in [(1.0, 1.0), (0.5, 1.0), (0.3, 0.5)]:
                w_jt = add_rows(pm, pos, w_pr, ev, rw, tp)
                for i in vis:
                    f = rows[i].feats
                    if f not in m['cidx']: continue
                    d_jt = w_jt @ m['P'][:, m['cidx'][f], :]
                    acc[f'joint_rw{rw}_t{tp}'].append(int(d_jt.argmax()) == gold[i])
                    sc = np.log(masses[i]) + args.lam * np.log(d_jt + 1e-9)
                    acc[f'comb_joint_rw{rw}_t{tp}'].append(int(sc.argmax()) == gold[i])
            for i in vis:
                f = rows[i].feats
                if f not in m['cidx']: continue
                d_pr = w_pr @ m['P'][:, m['cidx'][f], :]
                d_or = w_or @ m['P'][:, m['cidx'][f], :]
                acc['prior'].append(int(d_pr.argmax()) == gold[i])
                acc['nn'].append(int(np.argmax(masses[i])) == gold[i])
                acc['comb'].append(int((np.log(masses[i]) + args.lam * np.log(d_pr + 1e-9)).argmax()) == gold[i])
                acc['oracle_par'].append(int(d_or.argmax()) == gold[i])
print(f'segment={seg} lam={args.lam} K={args.K} reps={args.reps}')
for k in sorted(acc):
    print(f'  {k:24s} class acc {np.mean(acc[k]):.4f}  (n={len(acc[k])})')
