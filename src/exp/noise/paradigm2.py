"""Copy of src/paradigm.py ParadigmModel with a smoothing pseudo-count parameter (alpha)
and optional per-row weights (w<1 for suspected-noisy rows)."""
import numpy as np, pandas as pd, collections, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from paradigm import stress_class, NC


class ParadigmModel2:
    def __init__(self, K=300, iters=25, seed=0, alpha=1.0, prior_alpha=1.0):
        self.K, self.iters, self.seed, self.alpha, self.prior_alpha = K, iters, seed, alpha, prior_alpha
        self.models = {}

    def fit(self, df):
        df = df.copy()
        df['cls'] = df.form_vvz.map(stress_class)
        df = df[df.cls >= 0]
        rng = np.random.default_rng(self.seed)
        for pos, d in df.groupby('pos'):
            g = d.groupby(['lemma_ru', 'feats']).cls.agg(lambda s: collections.Counter(s).most_common(1)[0][0])
            x = g.unstack()
            cells = list(x.columns)
            A = x.fillna(-1).values.astype(int)
            L, C = A.shape
            K = self.K
            cents = np.array([[rng.integers(0, NC) if A[i, j] < 0 else A[i, j] for j in range(C)]
                              for i in rng.choice(L, K, replace=False)])
            for it in range(self.iters):
                dist = np.zeros((L, K))
                for k in range(K):
                    dist[:, k] = ((A != cents[k][None, :]) & (A >= 0)).sum(1)
                asg = dist.argmin(1)
                for k in range(K):
                    m = A[asg == k]
                    if len(m) == 0:
                        continue
                    for j in range(C):
                        v = m[:, j]; v = v[v >= 0]
                        if len(v):
                            cents[k, j] = np.bincount(v, minlength=NC).argmax()
            P = np.full((K, C, NC), self.alpha)
            for k in range(K):
                m = A[asg == k]
                for j in range(C):
                    v = m[:, j]; v = v[v >= 0]
                    P[k, j] += np.bincount(v, minlength=NC) if len(v) else 0
            P /= P.sum(-1, keepdims=True)
            prior = np.bincount(asg, minlength=K) + self.prior_alpha; prior /= prior.sum()
            self.models[pos] = dict(cells=cells, cidx={c: i for i, c in enumerate(cells)}, P=P, prior=prior, asg=asg, lemmas=list(x.index))
        return self

    def class_dist(self, pos, feats, observed):
        m = self.models[pos]
        if feats not in m['cidx']:
            return np.full(NC, 1.0 / NC)
        logp = np.log(m['prior']).copy()
        seen = {}
        for f, form in observed:
            if f not in m['cidx']:
                continue
            c = stress_class(form)
            if c < 0:
                continue
            seen.setdefault(f, []).append(c)
        for f, cs in seen.items():
            c = collections.Counter(cs).most_common(1)[0][0]
            logp += np.log(m['P'][:, m['cidx'][f], c])
        w = np.exp(logp - logp.max()); w /= w.sum()
        return w @ m['P'][:, m['cidx'][feats], :]
