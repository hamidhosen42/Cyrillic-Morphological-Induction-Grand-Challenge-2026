"""Local validation harness mimicking the test: segments + tier-weighted metric.

Segments (dev mode, seed 123 holdouts, same as the Kaggle/local training scripts):
  transfer  : dev rows whose (lemma, feats) exists in train (other dialect)   weight 1
  complete  : dev rows whose (lemma, feats) is unseen                          weight 2
  wug       : 500 held-out N/V lemmas, only their SEV seed cell known          weight 3
  unseen    : 250 held-out ADJ lemmas, nothing known                            weight 4
Score = 0.9 * weighted EM + 0.1 * (1 - weighted CER), CER = lev / len(gold) per row.
"""
import os, random, sys
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein as Lev
sys.path.insert(0, os.path.dirname(__file__))

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
W = {'transfer': 1.0, 'complete': 2.0, 'wug': 3.0, 'unseen': 4.0}
# test row counts per segment (for weighting the local segments like the test)
TEST_N = {'transfer': 18000, 'complete': 15000, 'wug': 18260, 'unseen': 8740}


def load_splits():
    tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
    wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); wug['lemma_frequency'] = 60.0
    rng = random.Random(123)
    lem_adj = sorted(set(tr[tr.pos == 'ADJ'].lemma_ru))
    seedrows = tr[((tr.feats == 'N;ACC;SG') | (tr.feats == 'V;PRS;3;SG')) & (tr.dialect == 'SEV')]
    hold_wug = set(rng.sample(sorted(set(seedrows.lemma_ru)), 500))
    hold_adj = set(rng.sample(lem_adj, 250))
    train_rows = tr[~tr.lemma_ru.isin(hold_wug | hold_adj)]
    lf = set(zip(train_rows.lemma_ru, train_rows.feats))
    dv = dv[~dv.lemma_ru.isin(hold_wug | hold_adj)]  # dev rows of held-out lemmas are not 'train-lemma' rows
    seen = np.array([(l, f) in lf for l, f in zip(dv.lemma_ru, dv.feats)])
    w = tr[tr.lemma_ru.isin(hold_wug)]
    wseed = w[((w.feats == 'N;ACC;SG') | (w.feats == 'V;PRS;3;SG')) & (w.dialect == 'SEV')]
    val = {'transfer': dv[seen], 'complete': dv[~seen], 'wug': w.drop(wseed.index), 'unseen': tr[tr.lemma_ru.isin(hold_adj)]}
    known = pd.concat([train_rows, wseed])
    return train_rows, known, val


def row_scores(preds, golds):
    em = np.array([p == g for p, g in zip(preds, golds)], dtype=float)
    cer = np.array([Lev.distance(p, g) / max(1, len(g)) for p, g in zip(preds, golds)])
    return em, cer


def weighted_score(seg_preds, seg_golds):
    """seg_preds/golds: dict segment -> list. Weighted like the test (row counts x tier weights)."""
    num_em = num_cer = den = 0.0
    per = {}
    for seg in seg_golds:
        em, cer = row_scores(seg_preds[seg], seg_golds[seg])
        per[seg] = (round(em.mean(), 4), round(cer.mean(), 4))
        wgt = W[seg] * TEST_N[seg]
        num_em += wgt * em.mean(); num_cer += wgt * cer.mean(); den += wgt
    score = 0.9 * num_em / den + 0.1 * (1 - num_cer / den)
    return round(score, 5), per
