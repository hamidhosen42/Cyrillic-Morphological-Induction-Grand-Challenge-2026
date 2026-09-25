"""(d): reflexive verb pairs X / Xся. On train pairs: per-cell stress-position agreement (stress index from start),
seed-cell agreement, and whether pooling the partner's seed / other cells improves prediction of X's cells."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
tr = pd.read_csv(f'{DATA}/train.csv'); wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv')
v = tr[tr.pos == 'V'].copy(); v['si'] = v.form_vvz.map(stress_idx); v['cls'] = v.form_vvz.map(stress_class); v = v[v.si >= 0]
lem = set(v.lemma_ru)
pairs = [(l, l + 'ся') for l in lem if l + 'ся' in lem]
print('train verb lemmas', len(lem), 'X/Xся pairs both in train', len(pairs))
wl = set(wug[wug.pos == 'V'].lemma_ru)
wpairs = [(l, l + 'ся') for l in wl if l + 'ся' in wl]
print('wug verb lemmas', len(wl), 'wug pairs both in wug', len(wpairs), 'wug verbs whose partner is in train', sum((l[:-2] in lem) if l.endswith('ся') else (l + 'ся' in lem) for l in wl))
# per-cell agreement of stress index (from start) between X and Xся, same (feats, dialect)
idx = {(r.lemma_ru, r.feats, r.dialect): (r.si, r.form_vvz) for r in v.itertuples()}
agree = collections.defaultdict(list); seed_agree = []; both = []
for x, xs in pairs:
    cells = [(f, d) for (l, f, d) in idx if l == x and (xs, f, d) in idx]
    for f, d in cells:
        a = idx[(x, f, d)][0] == idx[(xs, f, d)][0]
        agree[f].append(a)
        if f == 'V;PRS;3;SG' and d == 'SEV': seed_agree.append(a)
    both.append(len(cells))
allv = [a for f in agree for a in agree[f]]
print(f'cells compared {len(allv)}, stress-index agreement X vs Xся: {np.mean(allv):.3f}; seed cell (PRS;3;SG SEV) agreement {np.mean(seed_agree):.3f} n={len(seed_agree)}')
for f in sorted(agree): print(f'  {f:14s} agree {np.mean(agree[f]):.3f} n={len(agree[f])}')
# chance level: agreement between random unrelated verbs of same cell
rng = np.random.default_rng(0); ch = []
keys = list(idx)
for _ in range(20000):
    (l1, f, d) = keys[rng.integers(len(keys))]
    cand = [l2 for (l2, f2, d2) in (keys[j] for j in rng.integers(len(keys), size=50)) if f2 == f and d2 == d and l2 != l1]
    if cand: ch.append(idx[(l1, f, d)][0] == idx[(cand[0], f, d)][0])
print(f'chance agreement (random verb pairs, same cell) {np.mean(ch):.3f} n={len(ch)}')
shown = 0
for x, xs in pairs:
    for f in ['V;PRS;1;SG','V;PRS;3;SG','V;PRS;1;PL']:
        k1, k2 = (x, f, 'SEV'), (xs, f, 'SEV')
        if k1 in idx and k2 in idx and idx[k1][0] != idx[k2][0] and shown < 12:
            print('   disagree', f, idx[k1][1], idx[k2][1]); shown += 1
# conditional: given the seed cell agrees, agreement on other cells; given disagree
cond = {True: [], False: []}
for x, xs in pairs:
    if ('V;PRS;3;SG' in [f for (l, f, d) in idx if l == x]):
        pass
sidx = {}
for (l, f, d), (si, fm) in idx.items():
    if f == 'V;PRS;3;SG': sidx.setdefault(l, si)
for x, xs in pairs:
    if x in sidx and xs in sidx:
        sa = sidx[x] == sidx[xs]
        for (l, f, d) in list(idx):
            if l == x and f != 'V;PRS;3;SG' and (xs, f, d) in idx:
                cond[sa].append(idx[(x, f, d)][0] == idx[(xs, f, d)][0])
for k in cond: print(f'other-cell agreement given seed agree={k}: {np.mean(cond[k]) if cond[k] else float("nan"):.3f} n={len(cond[k])}')
# Practical value for wug pairs: both have only the seed cell. Does knowing partner's seed change P(target | seed)?
# Simulate on train pairs: predict X's cell class from lookup on (cell, seed_si, seed_nv, seed_end) with X's seed only vs. X's seed + partner-seed-agreement flag
from feats_ab_lib import featurize
train_rows, known, wug_df, seeds = get_splits()
trows, tseeds = train_seed_table(train_rows)
F = featurize(trows, tseeds); F['cls'] = trows.cls.values; F['lemma'] = trows.lemma_ru.values
partner = {}
for x, xs in pairs:
    if x in tseeds and xs in tseeds:
        partner[x] = tseeds[xs]; partner[xs] = tseeds[x]
F['has_partner'] = F.lemma.map(lambda l: l in partner)
F['p_agree'] = F.lemma.map(lambda l: (stress_idx(partner[l]) == stress_idx(tseeds[l])) if l in partner else -1)
sub = F[F.has_partner & (F.pos == 'V')]
print('train rows with partner seed available', len(sub), 'lemmas', sub.lemma.nunique())
from sklearn.model_selection import GroupKFold

class Lookup:
    def __init__(self, keysets, alpha=1.0): self.keysets, self.alpha = keysets, alpha
    def fit(self, F, y):
        self.tabs = []
        for ks in self.keysets:
            t = collections.defaultdict(lambda: np.zeros(NC))
            for key, c in zip(zip(*[F[k].values for k in ks]), y): t[key][c] += 1
            self.tabs.append(dict(t))
        return self
    def predict(self, F, min_n=20):
        out = np.zeros((len(F), NC)); cols = [list(zip(*[F[k].values for k in ks])) for ks in self.keysets]
        for i in range(len(F)):
            d = None
            for j in range(len(self.keysets) - 1, -1, -1):
                v = self.tabs[j].get(cols[j][i])
                if v is not None and v.sum() >= min_n: d = v; break
            if d is None: d = self.tabs[0].get(cols[0][i], np.ones(NC))
            out[i] = (d + self.alpha) / (d.sum() + self.alpha * NC)
        return out
base = ['feats', 'dialect', 'seed_cls']; k2 = base + ['seed_si', 'seed_nv', 'seed_end']
gkf = GroupKFold(5); y = sub.cls.values; r1 = np.zeros((len(sub), NC)); r2 = np.zeros((len(sub), NC))
for a, b in gkf.split(sub, groups=sub.lemma):
    # fit on ALL train rows except held-out lemmas (not just pair rows)
    hold = set(sub.lemma.iloc[b]); fit = F[~F.lemma.isin(hold) & (F.pos == 'V')]
    r1[b] = Lookup([base, k2]).fit(fit, fit.cls.values).predict(sub.iloc[b])
    fit2 = fit[fit.has_partner]
    r2[b] = Lookup([base, k2, k2 + ['p_agree']]).fit(fit2, fit2.cls.values).predict(sub.iloc[b], min_n=10)
print(f'pair-lemma rows: lookup(seed) acc {acc(r1, y):.4f} vs lookup(seed + partner-seed-agree flag) acc {acc(r2, y):.4f} n={len(y)}')
