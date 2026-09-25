"""Headroom decomposition on the seed-123 holdout using runs/kdev2/val_cands_aligned.pkl
(one Kaggle model trained without held-out lemmas, beam-8 candidates).
For each segment: top-1 EM, oracle EM (gold in beam), stress-oracle EM (top candidate of gold's
stress class), segmental accuracy (top-1 with accents stripped), and stress accuracy given
segmental correct.  Also the paradigm-prior-only selection (lam->inf) for reference.
"""
import os, sys, pickle, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from validate import load_splits, weighted_score
from paradigm import stress_class
ACUTE = '́'
strip = lambda s: s.replace(ACUTE, '')
train_rows, known, val = load_splits()
d = pickle.load(open(os.path.join(ROOT, 'runs/kdev2/val_cands_aligned.pkl'), 'rb'))
keymap = {'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}
shares = {'transfer': .131, 'complete': .218, 'wug': .398, 'unseen': .254}
rows = []
top1_preds, oracle_preds, stress_or_preds, golds = {}, {}, {}, {}
for seg, key in keymap.items():
    cands = d[key][0]; df = val[seg]
    assert len(cands) == len(df), (seg, len(cands), len(df))
    g = df.form_vvz.tolist()
    t1 = [c[0][0] if c else '' for c in cands]
    orc = [gold if any(f == gold for f, _ in c) else (c[0][0] if c else '') for c, gold in zip(cands, g)]
    sor = []
    for c, gold in zip(cands, g):
        gc = stress_class(gold)
        same = [f for f, _ in c if stress_class(f) == gc]
        sor.append(same[0] if same else (c[0][0] if c else ''))
    segok = np.array([strip(a) == strip(b) for a, b in zip(t1, g)])
    em = np.array([a == b for a, b in zip(t1, g)])
    rows.append(dict(segment=seg, n=len(g), top1_EM=em.mean(), oracle_beam8_EM=np.mean([a == b for a, b in zip(orc, g)]),
                     stress_oracle_EM=np.mean([a == b for a, b in zip(sor, g)]), segmental_acc_top1=segok.mean(),
                     stress_acc_given_seg_ok=em[segok].mean() if segok.any() else float('nan'),
                     gold_unaccented_in_beam=np.mean([any(strip(f) == strip(gold) for f, _ in c) for c, gold in zip(cands, g)])))
    top1_preds[seg], oracle_preds[seg], stress_or_preds[seg], golds[seg] = t1, orc, sor, g
res = pd.DataFrame(rows)
pd.set_option('display.width', 200); print(res.round(4).to_string(index=False))
for name, P in [('top1', top1_preds), ('stress_oracle', stress_or_preds), ('oracle_beam8', oracle_preds)]:
    s, per = weighted_score(P, golds); print(f'weighted score {name}: {s}')
# LB-points headroom per segment (0.9 * share * EM delta) from top1 -> stress oracle and -> full oracle
for r in rows:
    seg = r['segment']
    print(f"{seg:9s} headroom to stress-oracle: {0.9*shares[seg]*(r['stress_oracle_EM']-r['top1_EM']):+.4f} LB pts; to beam-oracle: {0.9*shares[seg]*(r['oracle_beam8_EM']-r['top1_EM']):+.4f} LB pts")
res.to_csv(os.path.join(ROOT, 'runs/exp/competitor-intel/headroom.csv'), index=False)

# per-segment (EM, CER) for top1 and stress-oracle, plus implied weighted EM of the leaders
for name, P in [('top1', top1_preds), ('stress_oracle', stress_or_preds)]:
    s, per = weighted_score(P, golds); print(name, 'per-seg (EM, CER):', {k: tuple(round(x, 4) for x in v) for k, v in per.items()})
# average CER of a wrong row (needed to convert LB score -> weighted EM)
allc = []
for seg in keymap:
    from rapidfuzz.distance import Levenshtein as Lev
    for p, g in zip(top1_preds[seg], golds[seg]):
        if p != g: allc.append(Lev.distance(p, g) / max(1, len(g)))
cer_wrong = float(np.mean(allc)); print(f'mean CER of a wrong top-1 row: {cer_wrong:.4f} (n={len(allc)})')
for lb in (0.66871, 0.69891, 0.69435):
    # score = 0.9*WEM + 0.1*(1 - (1-WEM)*cer_wrong)  -> WEM = (score - 0.1 + 0.1*cer_wrong) / (0.9 + 0.1*cer_wrong)
    wem = (lb - 0.1 + 0.1 * cer_wrong) / (0.9 + 0.1 * cer_wrong)
    print(f'LB {lb:.5f} -> implied weighted EM {wem:.4f}')
