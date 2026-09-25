"""Build declension-aware lookup priors for wug rows (train + wug holdout), tune smoothing, save."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from common import *
from feats_ab_lib import featurize
from sklearn.model_selection import GroupKFold
train_rows, known, wug_df, seeds = get_splits()
B = pickle.load(open(os.path.join(CACHE, 'baseline_wug.pkl'), 'rb')); pr_w, mass_w, gold_w = B['pr'], B['mass'], B['gold']
trows, tseeds = train_seed_table(train_rows); gold_t = trows.cls.values
pr_t = pickle.load(open(os.path.join(CACHE, 'train_fold_priors.pkl'), 'rb'))
Ft = featurize(trows, tseeds); Fw = featurize(wug_df, seeds)


class Lookup:
    def __init__(self, keysets, alpha=1.0, min_n=20): self.keysets, self.alpha, self.min_n = keysets, alpha, min_n
    def fit(self, F, y):
        self.tabs = []
        for ks in self.keysets:
            t = collections.defaultdict(lambda: np.zeros(NC))
            for key, c in zip(zip(*[F[k].values for k in ks]), y): t[key][c] += 1
            self.tabs.append(dict(t))
        return self
    def predict(self, F, mode='backoff'):
        out = np.zeros((len(F), NC)); cols = [list(zip(*[F[k].values for k in ks])) for ks in self.keysets]
        for i in range(len(F)):
            if mode == 'backoff':
                d = None
                for j in range(len(self.keysets) - 1, -1, -1):
                    v = self.tabs[j].get(cols[j][i])
                    if v is not None and v.sum() >= self.min_n: d = v; break
                if d is None: d = self.tabs[0].get(cols[0][i], np.ones(NC))
                out[i] = (d + self.alpha) / (d.sum() + self.alpha * NC)
            else:  # hierarchical shrinkage: each level smoothed towards the coarser level
                p = np.ones(NC) / NC
                for j in range(len(self.keysets)):
                    v = self.tabs[j].get(cols[j][i], np.zeros(NC))
                    p = (v + self.alpha * p) / (v.sum() + self.alpha)
                out[i] = p
        return out


base = ['feats', 'dialect', 'seed_cls']
K = {'A': [base, base + ['seed_end'], base + ['seed_si', 'seed_nv', 'seed_end']],
     'B': [base, base + ['seed_end'], base + ['seed_si', 'seed_nv', 'seed_end'], base + ['seed_si', 'seed_nv', 'lem_nv', 'seed_end']],
     'C': [base, base + ['seed_end'], base + ['seed_si', 'seed_nv', 'seed_end'], base + ['seed_si', 'seed_nv', 'seed_end', 'seed_end2']],
     'D': [base, base + ['seed_end'], base + ['seed_si', 'seed_nv', 'seed_end'], base + ['seed_si', 'seed_nv', 'seed_end', 'yat']]}
gkf = GroupKFold(5); ok = gold_w >= 0; best = None
for kn, ks in K.items():
    for mode, alpha, min_n in [('backoff', 1.0, 20), ('backoff', 1.0, 5), ('backoff', 0.5, 10), ('shrink', 5.0, 0), ('shrink', 20.0, 0), ('shrink', 50.0, 0)]:
        pt = np.zeros((len(Ft), NC))
        for a, b in gkf.split(Ft, groups=trows.lemma_ru):
            pt[b] = Lookup(ks, alpha, min_n).fit(Ft.iloc[a], gold_t[a]).predict(Ft.iloc[b], mode)
        pw = Lookup(ks, alpha, min_n).fit(Ft, gold_t).predict(Fw, mode)
        cv = acc(pt, gold_t); aw = acc(pw[ok], gold_w[ok]); cw = comb_acc(mass_w[ok], pw[ok], gold_w[ok])
        # log-likelihood as well (sharper metric than argmax)
        ll = float(np.mean(np.log(pt[np.arange(len(pt)), gold_t] + 1e-9)))
        print(f'keys {kn} {mode:8s} a={alpha} min_n={min_n}: train-CV acc {cv:.4f} ll {ll:.4f} | wug prior {aw:.4f} comb {cw:.4f}', flush=True)
        if best is None or cv > best[0]: best = (cv, kn, mode, alpha, min_n, pw, pt)
print('best by train-CV:', best[:5])
pickle.dump(dict(pw=best[5], pt=best[6], cfg=best[1:5]), open(os.path.join(CACHE, 'lt_prior.pkl'), 'wb'))
print(f'paradigm prior baseline: train-CV {acc(pr_t, gold_t):.4f} ll {np.mean(np.log(pr_t[np.arange(len(pr_t)), gold_t] + 1e-9)):.4f} | wug prior {acc(pr_w[ok], gold_w[ok]):.4f} comb {comb_acc(mass_w[ok], pr_w[ok], gold_w[ok]):.4f}')
