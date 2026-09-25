"""Holdout unseen-ADJ evaluation: baseline (candmerge base decision) vs nv-conditioned priors vs joint per-lemma.
Rows subsampled to 7 per lemma (R repeats) to mimic the test; results reweighted to test nv distribution."""
import sys, os, pickle, collections, argparse
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ParadigmModel, stress_class, n_vow, NC
from joint import combine_tally, build_masses, JointParadigm
from validate import load_splits
p = argparse.ArgumentParser(); p.add_argument('--R', type=int, default=5); p.add_argument('--K', type=int, default=300)
p.add_argument('--rows_per_lemma', type=int, default=7); p.add_argument('--full', action='store_true')
args = p.parse_args()
train_rows, known, val = load_splits()
vu = val['unseen'].reset_index(drop=True)
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl', 'rb'))['unseen_adj']
assert len(ext[0]) == len(vu)
tallies = [combine_tally(ext, i) for i in range(len(vu))]
masses = build_masses(tallies)
gold = list(vu.form_vvz); gcls = np.array([stress_class(f) for f in gold])
vu['nv'] = vu.lemma_ru.map(n_vow).clip(2, 6)
# test nv distribution for reweighting (per lemma)
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
teu = te[~te.lemma_ru.isin(set(tr.lemma_ru)) & ~te.lemma_ru.isin(set(wug.lemma_ru))].drop_duplicates('lemma_ru')
tnv = teu.lemma_ru.map(n_vow).clip(2, 6).value_counts(normalize=True).sort_index()
hl = vu.drop_duplicates('lemma_ru'); hnv = hl.nv.value_counts(normalize=True).sort_index()
print('holdout lemmas per nv', hl.nv.value_counts().sort_index().to_dict(), ' test nv frac', tnv.round(3).to_dict())
wrow = vu.nv.map(lambda v: tnv.get(v, 0) / hnv.get(v, 1e-9)).values
# priors
adj = train_rows[train_rows.pos == 'ADJ'].copy(); adj['cls'] = adj.form_vvz.map(stress_class); adj = adj[adj.cls >= 0]
g = adj.groupby(['lemma_ru', 'feats']).cls.agg(lambda s: collections.Counter(s).most_common(1)[0][0]).reset_index()
g['nv'] = g.lemma_ru.map(n_vow).clip(2, 6)
def table(keys, alpha=1.0):
    t = g.groupby(keys + ['cls']).size().unstack(fill_value=0).reindex(columns=range(NC), fill_value=0) + alpha
    return (t.T / t.sum(axis=1)).T
Pg = table(['feats']); Pnv = table(['feats', 'nv'])
pms = [ParadigmModel(K=args.K, seed=s).fit(train_rows) for s in (0, 1, 2)]
prior_pm = {f: sum(pm.class_dist('ADJ', f, []) for pm in pms) / 3 for f in vu.feats.unique()}
# per-nv paradigm models (fit on ADJ rows of that nv bucket only)
pms_nv = {}
for v in range(2, 7):
    d = train_rows[(train_rows.pos == 'ADJ') & (train_rows.lemma_ru.map(n_vow).clip(2, 6) == v)]
    nl = d.lemma_ru.nunique(); K = max(10, min(args.K, nl // 3))
    pms_nv[v] = [ParadigmModel(K=K, seed=s).fit(d) for s in (0, 1, 2)]
    print(f'nv{v}: {nl} lemmas, K={K}')
def pick(i, c):
    best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c), key=lambda x: x[1], default=None)
    return best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]

def decide_prior(idx, prior_fn, lam):
    """group by (lemma, feats); prior_fn(row) -> (NC,)"""
    groups = collections.defaultdict(list)
    for i in idx: groups[(vu.lemma_ru[i], vu.feats[i])].append(i)
    out = {}
    for (l, f), ii in groups.items():
        pr = prior_fn(ii[0])
        sc = sum(np.log(masses[i]) for i in ii) + lam * np.log(pr + 1e-9)
        c = int(sc.argmax())
        for i in ii: out[i] = pick(i, c)
    return out

def decide_joint(idx, models_fn, lam, temp=1.0, row_w=1.0):
    by_lemma = collections.defaultdict(list)
    for i in idx: by_lemma[vu.lemma_ru[i]].append(i)
    out = {}
    for l, ii in by_lemma.items():
        jp = JointParadigm(models_fn(ii[0]), temp=temp, row_w=row_w)
        cells = collections.defaultdict(list)
        for i in ii: cells[vu.feats[i]].append(i)
        cl = list(cells); cr = [(f, np.mean([masses[i] for i in cells[f]], axis=0)) for f in cl]
        dists = jp.class_dists('ADJ', [], cr)
        for f, d in zip(cl, dists):
            for i in cells[f]:
                sc = np.log(masses[i]) + lam * np.log(d + 1e-9)
                out[i] = pick(i, int(sc.argmax()))
    return out

def em(out, idx):
    e = np.array([out[i] == gold[i] for i in idx], dtype=float); w = wrow[idx]
    return e.mean(), np.average(e, weights=w)

rng = np.random.default_rng(0)
by_lemma = vu.groupby('lemma_ru').indices
configs = collections.OrderedDict()
configs['nn_only'] = lambda idx: {i: max(tallies[i].items(), key=lambda x: x[1])[0] for i in idx}
configs['oracle_cls'] = lambda idx: {i: pick(i, gcls[i]) for i in idx}
configs['base_pm_lam2'] = lambda idx: decide_prior(idx, lambda i: prior_pm[vu.feats[i]], 2.0)
configs['base_pm_lam1'] = lambda idx: decide_prior(idx, lambda i: prior_pm[vu.feats[i]], 1.0)
configs['glob_tab_lam2'] = lambda idx: decide_prior(idx, lambda i: Pg.loc[vu.feats[i]].values, 2.0)
for lam in (1.0, 2.0, 3.0):
    configs[f'nv_tab_lam{lam}'] = (lambda lam: lambda idx: decide_prior(idx, lambda i: Pnv.loc[(vu.feats[i], vu.nv[i])].values, lam))(lam)
configs['joint_glob_lam2'] = lambda idx: decide_joint(idx, lambda i: pms, 2.0)
configs['joint_glob_lam1'] = lambda idx: decide_joint(idx, lambda i: pms, 1.0)
for lam in (1.0, 2.0, 3.0):
    configs[f'joint_nv_lam{lam}'] = (lambda lam: lambda idx: decide_joint(idx, lambda i: pms_nv[vu.nv[i]], lam))(lam)
configs['joint_nv_lam2_t0.5'] = lambda idx: decide_joint(idx, lambda i: pms_nv[vu.nv[i]], 2.0, temp=0.5)
configs['joint_nv_lam2_rw0.5'] = lambda idx: decide_joint(idx, lambda i: pms_nv[vu.nv[i]], 2.0, row_w=0.5)
configs['joint_nv_lam2_rw0'] = lambda idx: decide_joint(idx, lambda i: pms_nv[vu.nv[i]], 2.0, row_w=0.0)
res = collections.defaultdict(list); res_nv = collections.defaultdict(lambda: collections.defaultdict(list))
reps = 1 if args.full else args.R
for r in range(reps):
    if args.full: idx = np.arange(len(vu))
    else: idx = np.concatenate([rng.choice(ii, min(args.rows_per_lemma, len(ii)), replace=False) for ii in by_lemma.values()])
    for name, fn in configs.items():
        out = fn(idx); res[name].append(em(out, idx))
        for v in range(2, 7):
            sub = idx[vu.nv.values[idx] == v]
            if len(sub): res_nv[name][v].append(np.mean([out[i] == gold[i] for i in sub]))
print(f'n rows per rep = {len(idx)}, reps = {reps}')
print(f'{"config":22s} {"EM":>7s} {"EM_testnv":>9s}  ' + ' '.join(f'{"nv"+str(v):>6s}' for v in range(2, 7)))
for name in configs:
    a = np.array(res[name])
    print(f'{name:22s} {a[:, 0].mean():.4f} {a[:, 1].mean():9.4f}  ' + ' '.join(f'{np.mean(res_nv[name][v]):6.3f}' for v in range(2, 7)))
