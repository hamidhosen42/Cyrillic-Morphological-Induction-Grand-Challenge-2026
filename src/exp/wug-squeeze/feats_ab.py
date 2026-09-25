"""(a)+(b): does seed stress index / vowel counts / seed ending (declension) predict the target class beyond the paradigm prior?
Lookup tables with backoff + GBM stacking. Train on train lemmas with SEV seed (GroupKFold priors), eval on wug holdout rows."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

train_rows, known, wug_df, seeds = get_splits()
B = pickle.load(open(os.path.join(CACHE, 'baseline_wug.pkl'), 'rb'))
pr_w, mass_w, gold_w = B['pr'], B['mass'], B['gold']
trows, tseeds = train_seed_table(train_rows)
print('train seed rows', len(trows), 'lemmas', trows.lemma_ru.nunique())

# ---- GroupKFold priors for train rows (paradigm models fit without the fold's lemmas)
fn = os.path.join(CACHE, 'train_fold_priors.pkl')
if os.path.exists(fn):
    pr_t = pickle.load(open(fn, 'rb'))
else:
    pr_t = np.zeros((len(trows), NC)); gkf = GroupKFold(5)
    for k, (a, b) in enumerate(gkf.split(trows, groups=trows.lemma_ru)):
        hold = set(trows.lemma_ru.iloc[b]); fit = train_rows[~train_rows.lemma_ru.isin(hold)]
        pms = paradigm_models(fit, tag=f'fold{k}')
        pr_t[b] = prior_dists(pms, trows.iloc[b], tseeds)
        print('fold', k, 'prior acc', acc(pr_t[b], trows.cls.values[b]), flush=True)
    pickle.dump(pr_t, open(fn, 'wb'))
gold_t = trows.cls.values
print(f'train-CV prior acc {acc(pr_t, gold_t):.4f} n={len(gold_t)}')


def endtype(seed, pos):
    p = plain(seed)
    if pos == 'N':
        last = p[-1]
        return 'V:' + last if last in VOW else ('J:' + last if last in 'ъь' else 'C')
    # verb: theme vowel before the last consonant cluster
    vs = [c for c in p if c in VOW]
    return p[-3:] + '|' + (vs[-1] if vs else '')


def featurize(df, seedmap):
    F = pd.DataFrame(index=df.index)
    sd = df.lemma_ru.map(seedmap)
    F['feats'] = df.feats.values; F['dialect'] = df.dialect.values; F['pos'] = df.pos.values
    F['seed_cls'] = sd.map(stress_class).values
    F['seed_si'] = sd.map(stress_idx).values
    F['seed_nv'] = sd.map(n_vow).values
    F['lem_nv'] = df.lemma_ru.map(n_vow).values
    F['dnv'] = F.seed_nv - F.lem_nv
    F['seed_end'] = [endtype(s, p) for s, p in zip(sd, df.pos)]
    F['seed_end2'] = sd.map(lambda s: plain(s)[-2:]).values
    F['lem_end2'] = df.lemma_ru.str[-2:].values
    F['yat'] = df.yat_flag.values
    F['seed_from_end'] = F.seed_nv - 1 - F.seed_si
    return F


Ft = featurize(trows, tseeds); Fw = featurize(wug_df, seeds)


class Lookup:
    """P(cls | key) with backoff to coarser keys, additive smoothing."""
    def __init__(self, keysets, alpha=1.0): self.keysets, self.alpha = keysets, alpha
    def fit(self, F, y):
        self.tabs = []
        for ks in self.keysets:
            t = collections.defaultdict(lambda: np.zeros(NC))
            for key, c in zip(zip(*[F[k].values for k in ks]), y): t[key][c] += 1
            self.tabs.append(dict(t))
        return self
    def predict(self, F, min_n=20):
        out = np.zeros((len(F), NC))
        cols = [list(zip(*[F[k].values for k in ks])) for ks in self.keysets]
        for i in range(len(F)):
            d = None
            for j in range(len(self.keysets) - 1, -1, -1):  # most specific first
                v = self.tabs[j].get(cols[j][i])
                if v is not None and v.sum() >= min_n:
                    d = v; break
            if d is None: d = self.tabs[0].get(cols[0][i], np.ones(NC))
            out[i] = (d + self.alpha) / (d.sum() + self.alpha * NC)
        return out


ok = gold_w >= 0
res = {}
def report(name, pw, pt=None):
    a = acc(pw[ok], gold_w[ok]); c = comb_acc(mass_w[ok], pw[ok], gold_w[ok])
    s = f'{name:34s} wug prior-acc {a:.4f} comb-acc {c:.4f}'
    if pt is not None: s += f' | train-CV acc {acc(pt, gold_t):.4f}'
    print(s, flush=True); res[name] = (a, c)

report('paradigm prior (baseline)', pr_w, pr_t)
base = ['feats', 'dialect', 'seed_cls']
configs = {
    'LT seed_cls': [base],
    'LT +dnv': [base, base + ['dnv']],
    'LT +seed_end': [base, base + ['seed_end']],
    'LT +seed_end+dnv': [base, base + ['seed_end'], base + ['seed_end', 'dnv']],
    'LT +seed_si+seed_nv': [base, base + ['seed_si', 'seed_nv']],
    'LT +si+nv+lem_nv': [base, base + ['seed_si', 'seed_nv'], base + ['seed_si', 'seed_nv', 'lem_nv']],
    'LT +si+nv+seed_end': [base, base + ['seed_si', 'seed_nv'], base + ['seed_si', 'seed_nv', 'seed_end']],
    'LT +si+nv+lem_nv+seed_end': [base, base + ['seed_si', 'seed_nv'], base + ['seed_si', 'seed_nv', 'seed_end'], base + ['seed_si', 'seed_nv', 'lem_nv', 'seed_end']],
    'LT +seed_end2': [base, base + ['seed_end'], base + ['seed_end2']],
}
# train-CV for lookups: GroupKFold over lemmas
gkf = GroupKFold(5)
for name, ks in configs.items():
    lt = Lookup(ks).fit(Ft, gold_t); pw = lt.predict(Fw)
    pt = np.zeros((len(Ft), NC))
    for a, b in gkf.split(Ft, groups=trows.lemma_ru):
        pt[b] = Lookup(ks).fit(Ft.iloc[a], gold_t[a]).predict(Ft.iloc[b])
    report(name, pw, pt)

# ---- GBM stacking: prior dist + features
def gbm_frame(F, pr, use_prior=True, cats=None):
    X = pd.DataFrame(index=F.index)
    catcols = ['feats', 'dialect', 'seed_end', 'seed_end2', 'lem_end2']
    for c in catcols:
        m = cats[c]; X[c] = F[c].map(lambda v: m.get(v, 0)).astype(int).values
    for c in ['seed_cls', 'seed_si', 'seed_nv', 'lem_nv', 'dnv', 'yat', 'seed_from_end']: X[c] = F[c].values
    if use_prior:
        for j in range(NC): X[f'lp{j}'] = np.log(pr[:, j] + 1e-6)
    return X, [list(X.columns).index(c) for c in catcols]

cats = {c: {v: i + 1 for i, v in enumerate(Ft[c].value_counts().index[:200])} for c in ['feats', 'dialect', 'seed_end', 'seed_end2', 'lem_end2']}
for use_prior in [True, False]:
    Xt, ci = gbm_frame(Ft, pr_t, use_prior, cats); Xw, _ = gbm_frame(Fw, pr_w, use_prior, cats)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, categorical_features=ci, random_state=0)
    clf.fit(Xt.values, gold_t)
    pw = np.full((len(Xw), NC), 1e-4); p = clf.predict_proba(Xw.values)
    for j, c in enumerate(clf.classes_): pw[:, int(c)] = p[:, j]
    pw /= pw.sum(1, keepdims=True)
    # train CV
    pt = np.zeros((len(Xt), NC))
    for a, b in gkf.split(Xt, groups=trows.lemma_ru):
        cl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, categorical_features=ci, random_state=0).fit(Xt.values[a], gold_t[a])
        p = cl.predict_proba(Xt.values[b]); q = np.full((len(b), NC), 1e-4)
        for j, c in enumerate(cl.classes_): q[:, int(c)] = p[:, j]
        pt[b] = q / q.sum(1, keepdims=True)
    report(f'GBM {"prior+" if use_prior else ""}all feats', pw, pt)
    if use_prior:
        imp = sorted(zip(clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else Xt.columns, np.zeros(len(Xt.columns))), key=lambda x: -x[1])
pickle.dump(res, open(os.path.join(CACHE, 'feats_ab_res.pkl'), 'wb'))
