"""Label-free check on TEST unseen rows: expected class accuracy of the decided class under train P(c | nv, cell),
for base (pm prior) vs nv-conditioned prior; calibrated against the holdout where real class accuracy is known."""
import sys, os, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from paradigm import ParadigmModel, stress_class, n_vow, NC
from joint import combine_tally, build_masses
from validate import load_splits
def class_table(df):
    adj = df[df.pos == 'ADJ'].copy(); adj['cls'] = adj.form_vvz.map(stress_class); adj = adj[adj.cls >= 0]
    g = adj.groupby(['lemma_ru', 'feats']).cls.agg(lambda s: collections.Counter(s).most_common(1)[0][0]).reset_index()
    g['nv'] = g.lemma_ru.map(n_vow).clip(2, 6)
    t = g.groupby(['feats', 'nv', 'cls']).size().unstack(fill_value=0).reindex(columns=range(NC), fill_value=0) + 1.0
    return (t.T / t.sum(axis=1)).T
def run(df, cpm, pms, Pnv, tag):
    rows = list(df.itertuples()); tallies = [combine_tally(cpm, i) for i in range(len(rows))]; masses = build_masses(tallies)
    prior_pm = {f: sum(pm.class_dist('ADJ', f, []) for pm in pms) / len(pms) for f in df.feats.unique()}
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    dec = {'base': np.zeros(len(rows), int), 'nvprior': np.zeros(len(rows), int), 'prior_only': np.zeros(len(rows), int)}
    for (lemma, feats), idx in groups.items():
        nv = min(max(n_vow(lemma), 2), 6); pn = Pnv.loc[(feats, nv)].values
        lm = sum(np.log(masses[i]) for i in idx)
        for i in idx:
            dec['base'][i] = int((lm + 2.0 * np.log(prior_pm[feats] + 1e-9)).argmax())
            dec['nvprior'][i] = int((lm + 2.0 * np.log(pn + 1e-9)).argmax())
            dec['prior_only'][i] = int(pn.argmax())
    nvs = np.array([min(max(n_vow(r.lemma_ru), 2), 6) for r in rows])
    gc = np.array([stress_class(f) for f in df.form_vvz]) if 'form_vvz' in df.columns else None
    for name, d in dec.items():
        exp_acc = np.array([Pnv.loc[(r.feats, nvs[i])].values[d[i]] for i, r in enumerate(rows)])
        line = f'{tag} {name:10s} expected class acc (train stats) {exp_acc.mean():.4f}  by nv: ' + ' '.join(f'nv{v}:{exp_acc[nvs == v].mean():.3f}' for v in range(2, 7))
        if gc is not None:
            ok = (d == gc)[gc >= 0]; line += f' | REAL class acc {ok.mean():.4f} by nv: ' + ' '.join(f'nv{v}:{(d == gc)[(gc >= 0) & (nvs == v)].mean():.3f}' for v in range(2, 7))
        print(line, flush=True)
    d = dec['base']; d2 = dec['nvprior']; print(f'{tag} rows where nvprior changes class: {(d != d2).sum()} / {len(rows)}; nv2 share of rows {np.mean(nvs == 2):.3f}')
    # class-0 share on NOM;SG;SHORT nv2 (train truth 0.99)
    m = np.array([(r.feats == 'ADJ;NOM;SG;SHORT') and nvs[i] == 2 for i, r in enumerate(rows)])
    if m.sum(): print(f'{tag} NOM;SG;SHORT nv2 rows n={m.sum()}: class0 share base {(d[m] == 0).mean():.3f} nvprior {(d2[m] == 0).mean():.3f} (train truth 0.99)')
    m = np.array([(r.feats.endswith('LONG')) and nvs[i] == 2 for i, r in enumerate(rows)])
    print(f'{tag} LONG nv2 rows n={m.sum()}: class0 share base {(d[m] == 0).mean():.3f} nvprior {(d2[m] == 0).mean():.3f} (train truth ~0.72-0.77)')
# holdout calibration
train_rows, known, val = load_splits()
pms = [ParadigmModel(K=300, seed=s).fit(train_rows) for s in (0, 1, 2)]
ext = pickle.load(open('runs/kdev2/val_cands_aligned.pkl', 'rb'))['unseen_adj']
run(val['unseen'].reset_index(drop=True), ext, pms, class_table(train_rows), 'HOLDOUT')
# test
tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv'); te = pd.read_csv('data/test_features.csv'); wug = pd.read_csv('data/wug_seeds.csv')
full = pd.concat([tr, dv]); pms_f = [ParadigmModel(K=300, seed=s).fit(full) for s in (0, 1, 2)]
ext_te = pickle.load(open('runs/test_cands_v1v3.pkl', 'rb'))
mask = (~te.lemma_ru.isin(set(tr.lemma_ru)) & ~te.lemma_ru.isin(set(wug.lemma_ru))).values
idx = np.where(mask)[0]; teu = te[mask].reset_index(drop=True)
cpm = [[m[i] for i in idx] for m in ext_te]
run(teu, cpm, pms_f, class_table(full), 'TEST')
