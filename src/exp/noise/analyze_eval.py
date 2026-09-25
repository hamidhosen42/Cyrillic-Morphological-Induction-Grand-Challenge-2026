"""Break down holdout EM of an eval pickle by gold-noise status, lemma noise status, seed noise, dialect;
re-weight by the test dialect mix to estimate the test EM under the clean-test hypothesis."""
import sys, os, pickle, collections
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), '..', '..'); sys.path.insert(0, ROOT)
from validate import load_splits, W, TEST_N
from paradigm import stress_class
tag = sys.argv[1] if len(sys.argv) > 1 else 'base'
ev = pickle.load(open(f'runs/exp/noise/eval_{tag}.pkl', 'rb'))
train_rows, known, val = load_splits()
tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv'); te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
nz = pd.read_csv('runs/exp/noise/train_noise.csv'); nzd = pd.read_csv('runs/exp/noise/dev_noise2.csv')
noisy = set(tr.lemma_ru[nz.noisylemma.values]); subid = set(tr.id[nz.sub2.values]); stressmin = set(tr.id[nz.stressx.values == 2]); typoid = set(tr.id[nz.typo.values])
dsub = set(dv.id[nzd.sub2.values])
seedrows = tr[((tr.feats == 'N;ACC;SG') | (tr.feats == 'V;PRS;3;SG')) & (tr.dialect == 'SEV')]
seed_sub_lemmas = set(seedrows.lemma_ru[seedrows.id.isin(subid)])
wl = set(wug.lemma_ru); trl = set(tr.lemma_ru)
te_seg = np.where(te.lemma_ru.isin(trl), 'train', np.where(te.lemma_ru.isin(wl), 'wug', 'unseen'))
test_dial = {s: te[te_seg == s].dialect.value_counts(normalize=True).to_dict() for s in ['train', 'wug', 'unseen']}
tot = {}
for seg in ['transfer', 'complete', 'wug', 'unseen']:
    df = val[seg].reset_index(drop=True); pr = np.array(ev['preds'][seg]); gd = np.array(ev['golds'][seg])
    em = (pr == gd)
    cls_ok = np.array([stress_class(a) == stress_class(b) for a, b in zip(pr, gd)])
    if seg in ('wug', 'unseen'):
        gsub = df.id.isin(subid).values; gstress = df.id.isin(stressmin).values; gtypo = df.id.isin(typoid).values
    else:
        gsub = df.id.isin(dsub).values; gstress = np.zeros(len(df), bool); gtypo = np.zeros(len(df), bool)
    lnoisy = df.lemma_ru.isin(noisy).values; sseed = df.lemma_ru.isin(seed_sub_lemmas).values
    clean = ~gsub & ~gstress & ~gtypo
    ctx_clean = clean & ~lnoisy & ~sseed
    print(f'== {seg} n={len(df)} EM {em.mean():.4f} cls-ok {cls_ok.mean():.4f} | gold sub {gsub.mean():.4f} (EM {em[gsub].mean() if gsub.any() else float("nan"):.3f}) stress-min {gstress.mean():.4f} typo {gtypo.mean():.4f} noisy-lemma {lnoisy.mean():.4f} (EM {em[lnoisy].mean():.3f}) seed-sub {sseed.mean():.4f} (EM {em[sseed].mean() if sseed.any() else float("nan"):.3f})')
    print(f'   EM clean-gold {em[clean].mean():.4f} (n={clean.sum()})  EM clean-gold&clean-lemma&clean-seed {em[ctx_clean].mean():.4f} (n={ctx_clean.sum()})  errors with noisy gold: {(~em & ~clean).sum()}/{(~em).sum()} = {(~em & ~clean).sum() / max(1, (~em).sum()):.3f}')
    # segmental vs class errors on clean rows
    seg_err = (~em & cls_ok & ctx_clean).sum(); cls_err = (~cls_ok & ctx_clean).sum()
    print(f'   on clean rows: class errors {cls_err} ({cls_err / ctx_clean.sum():.4f}), segmental-only errors {seg_err} ({seg_err / ctx_clean.sum():.4f})')
    by_d = {d: (em[(df.dialect == d).values & ctx_clean].mean(), ((df.dialect == d).values & ctx_clean).sum()) for d in ['SEV', 'POM', 'KAM']}
    print('   clean EM by dialect:', {d: (round(v[0], 4), v[1]) for d, v in by_d.items()})
    tseg = 'train' if seg in ('transfer', 'complete') else seg
    rew = sum(test_dial[tseg][d] * by_d[d][0] for d in by_d)
    rew_all = sum(test_dial[tseg][d] * em[(df.dialect == d).values].mean() for d in by_d)
    print(f'   test-dialect-reweighted EM: all-rows {rew_all:.4f}  clean-rows {rew:.4f}   (test dialect mix {dict((k, round(v, 3)) for k, v in test_dial[tseg].items())})')
    tot[seg] = dict(all=em.mean(), rew_all=rew_all, rew_clean=rew, clean=em[ctx_clean].mean())
den = sum(W[s] * TEST_N[s] for s in tot)
for key in ['all', 'rew_all', 'rew_clean']:
    print(f'weighted EM ({key}): {sum(W[s] * TEST_N[s] * tot[s][key] for s in tot) / den:.4f}')
