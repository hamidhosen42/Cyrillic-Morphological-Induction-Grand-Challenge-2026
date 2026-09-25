"""Phase A: simple structure tests of yat_flag and paradigm labels along orderings."""
import pandas as pd, numpy as np, pickle, sys
from scipy import stats
L = pd.read_pickle('runs/exp/rng_paradigm/lemmas.pkl').sort_values('rank').reset_index(drop=True)
lab = pickle.load(open('runs/exp/rng_paradigm/labels_k64.pkl', 'rb'))
# attach K=64 label (seed0), only confident (mism<=1)
L['lab'] = -1
for pos, d in lab.items():
    m = dict(zip(d['lemmas'], np.where(d['mism'] <= 1, d['labels'][0], -1)))
    idx = L.pos == pos
    L.loc[idx, 'lab'] = L.loc[idx, 'lemma_ru'].map(m).fillna(-1).astype(int)
print('labelled lemmas:', (L.lab >= 0).sum())
y = L.yat_flag.values
print('yat count', y.sum(), 'of', len(y), 'rate', y.mean())
# runs test on yat
runs = 1 + (y[1:] != y[:-1]).sum(); n1, n0 = y.sum(), len(y) - y.sum()
mu = 1 + 2 * n1 * n0 / (n1 + n0); var = 2 * n1 * n0 * (2 * n1 * n0 - n1 - n0) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
print('yat runs z =', (runs - mu) / np.sqrt(var))
# autocorrelation of yat, lags 1..50
yc = y - y.mean()
ac = [np.corrcoef(yc[:-k], yc[k:])[0, 1] for k in range(1, 51)]
print('yat max |autocorr| lag1..50:', np.max(np.abs(ac)).round(4), 'z=', (np.max(np.abs(ac)) * np.sqrt(len(y))).round(2), 'at lag', int(np.argmax(np.abs(ac)) + 1))
# yat rate vs rank mod m, chi2 with Bonferroni over m=2..200
best = []
for m in range(2, 201):
    tab = pd.crosstab(L['rank'] % m, y)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    best.append((p, m, chi2))
best.sort(); print('yat periodicity: best p (raw)', best[0], 'Bonferroni p', min(1, best[0][0] * 199))
# yat rate in blocks of 1000 along rank
blk = pd.Series(y).groupby(np.arange(len(y)) // 1000).mean().round(3).tolist(); print('yat rate per 1000-block', blk)
# yat vs pos, vs src
print(L.groupby('pos').yat_flag.mean().round(3).to_dict(), L.groupby('src').yat_flag.mean().round(3).to_dict())
# labels: within POS along rank, adjacent-equality rate vs expectation
for pos in ['ADJ', 'V', 'N']:
    g = L[(L.pos == pos) & (L.lab >= 0)]
    l = g.lab.values
    same = (l[1:] == l[:-1]).mean()
    p = np.bincount(l) / len(l); exp = (p ** 2).sum()
    n = len(l) - 1
    print(pos, 'n', len(l), 'adjacent same-label rate', round(same, 4), 'expected', round(exp, 4), 'z', round((same - exp) / np.sqrt(exp * (1 - exp) / n), 2))
    # lag-k same-label for k up to 20
    zs = []
    for k in range(1, 21):
        s = (l[k:] == l[:-k]).mean(); zs.append((s - exp) / np.sqrt(exp * (1 - exp) / (len(l) - k)))
    print('   lag z (1..20):', np.round(zs, 1))
    # label vs rank mod m: chi2 p-values, Bonferroni
    bb = []
    for m in range(2, 201):
        tab = pd.crosstab(g['rank'] % m, l)
        if (tab.values.sum(0) == 0).any(): continue
        chi2, p_, dof, _ = stats.chi2_contingency(tab)
        bb.append((p_, m))
    bb.sort(); print('   label periodicity best raw p', bb[0], 'bonf', min(1, bb[0][0] * 199))
    # label vs yat: independence
    tab = pd.crosstab(g.yat_flag, l); print('   label~yat chi2 p', stats.chi2_contingency(tab)[1])
    # label vs alphabetical neighbours
    ga = g.sort_values('lemma_ru'); la = ga.lab.values
    same = (la[1:] == la[:-1]).mean(); print('   alphabetical adjacent same rate', round(same, 4), 'z', round((same - exp) / np.sqrt(exp * (1 - exp) / n), 2))
    # label vs src of neighbouring wug/unseen: does the label depend on position parity etc: rank parity
    print('   label~rank parity p', stats.chi2_contingency(pd.crosstab(g['rank'] % 2, l))[1])
L.to_pickle('runs/exp/rng_paradigm/lemmas_lab.pkl')
