"""Shared harness: wug holdout rows, seeds, baseline paradigm prior + NN masses."""
import os, sys, pickle, collections, time
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from paradigm import ParadigmModel, stress_class, stress_idx, n_vow, NC, ACC, VOW
from joint import combine_tally, build_masses
from validate import load_splits, DATA

SEED_F = {'N': 'N;ACC;SG', 'V': 'V;PRS;3;SG'}
CACHE = os.path.join(ROOT, 'runs', 'exp', 'wug-squeeze')


def plain(s):
    return s.replace(ACC, '')


def get_splits():
    train_rows, known, val = load_splits()
    wug_df = val['wug'].reset_index(drop=True)
    seeds = {}
    for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
        if d == 'SEV' and f == SEED_F.get(l and known_pos(known, l), None):
            pass
    # seeds = rows of known that are the SEV seed cell for wug lemmas
    wl = set(wug_df.lemma_ru)
    ks = known[known.lemma_ru.isin(wl)]
    for r in ks.itertuples():
        seeds[r.lemma_ru] = r.form_vvz
    return train_rows, known, wug_df, seeds


def known_pos(known, l):
    return None


def train_seed_table(train_rows):
    """train lemmas (N/V) with SEV seed cell present -> seed form; rows = other cells."""
    t = train_rows[train_rows.pos.isin(['N', 'V'])]
    sd = t[(t.dialect == 'SEV') & (t.feats == t.pos.map(SEED_F))].drop_duplicates('lemma_ru')
    seeds = dict(zip(sd.lemma_ru, sd.form_vvz))
    rows = t[t.lemma_ru.isin(seeds) & (t.feats != t.pos.map(SEED_F))].copy()
    rows['cls'] = rows.form_vvz.map(stress_class)
    rows = rows[rows.cls >= 0].reset_index(drop=True)
    rows['seed'] = rows.lemma_ru.map(seeds)
    return rows, seeds


def paradigm_models(train_rows, K=300, seeds=(0, 1, 2), tag='base'):
    fn = os.path.join(CACHE, f'pms_{tag}_K{K}.pkl')
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    t0 = time.time()
    pms = [ParadigmModel(K=K, seed=s).fit(train_rows) for s in seeds]
    print(f'fit {len(seeds)} paradigm models in {time.time() - t0:.0f}s', flush=True)
    pickle.dump(pms, open(fn, 'wb'))
    return pms


def prior_dists(pms, df, seeds):
    out = np.zeros((len(df), NC))
    for i, r in enumerate(df.itertuples()):
        obs = [(SEED_F[r.pos], seeds[r.lemma_ru])]
        out[i] = sum(pm.class_dist(r.pos, r.feats, obs) for pm in pms) / len(pms)
    return out


def wug_masses(wug_df):
    ext = pickle.load(open(os.path.join(ROOT, 'runs', 'kdev2', 'val_cands_aligned.pkl'), 'rb'))
    cpm = ext['wug']
    assert len(cpm[0]) == len(wug_df), (len(cpm[0]), len(wug_df))
    tallies = [combine_tally(cpm, i) for i in range(len(wug_df))]
    return np.array(build_masses(tallies)), tallies


def acc(dist, gold):
    return float(np.mean(dist.argmax(1) == gold))


def comb_acc(mass, prior, gold, lam=2.0):
    sc = np.log(mass) + lam * np.log(prior + 1e-9)
    return float(np.mean(sc.argmax(1) == gold))
