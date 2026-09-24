"""Latent-class mixture model of stress paradigms, fitted by EM.

P(classes of a lemma) = Σ_k π_k Π_cells P(c_cell | k, cell)
Soft EM with Dirichlet smoothing; handles missing cells. Gives calibrated posteriors, which is what
matters when only ONE cell is observed (wug seed).
"""
import collections
import numpy as np
from paradigm import stress_class, NC


class MixtureParadigm:
    def __init__(self, K=60, iters=120, alpha=0.2, seed=0, tol=1e-5):
        self.K, self.iters, self.alpha, self.seed, self.tol = K, iters, alpha, seed, tol
        self.models = {}

    def fit(self, df, verbose=False):
        """df: rows with lemma_ru, pos, feats, form_vvz."""
        df = df.copy()
        df['cls'] = df.form_vvz.map(stress_class)
        df = df[df.cls >= 0]
        for pos, d in df.groupby('pos'):
            g = d.groupby(['lemma_ru', 'feats']).cls.agg(lambda s: collections.Counter(s).most_common(1)[0][0])
            x = g.unstack()
            cells = list(x.columns)
            A = x.fillna(-1).values.astype(int)          # (L, C)
            L, C = A.shape
            rng = np.random.default_rng(self.seed)
            # one-hot with missing mask
            obs = A >= 0
            X = np.zeros((L, C, NC))
            ii, jj = np.nonzero(obs)
            X[ii, jj, A[ii, jj]] = 1.0
            # init from random lemmas
            P = np.full((self.K, C, NC), 1.0 / NC)
            for k in range(self.K):
                idx = rng.choice(L, size=max(8, L // (4 * self.K)), replace=False)
                cnt = X[idx].sum(0) + self.alpha
                P[k] = cnt / cnt.sum(-1, keepdims=True)
            P = P * (1 - 0.15) + 0.15 * rng.dirichlet(np.ones(NC), size=(self.K, C))
            pi = np.full(self.K, 1.0 / self.K)
            prev = -np.inf
            logX = None
            for it in range(self.iters):
                # E-step: log P(lemma | k) = Σ_cells Σ_c X * log P
                logP = np.log(P + 1e-12)                                     # (K, C, NC)
                ll = np.tensordot(X, logP, axes=([1, 2], [1, 2]))            # (L, K)
                ll = ll + np.log(pi + 1e-12)
                mx = ll.max(1, keepdims=True)
                R = np.exp(ll - mx); s = R.sum(1, keepdims=True); R = R / s
                total = float((np.log(s) + mx).sum())
                # M-step
                pi = R.sum(0) + 1e-6; pi /= pi.sum()
                num = np.tensordot(R, X, axes=([0], [0]))                    # (K, C, NC)
                num = num + self.alpha
                P = num / num.sum(-1, keepdims=True)
                if abs(total - prev) < self.tol * abs(total):
                    break
                prev = total
            if verbose:
                print(f'  {pos}: L={L} C={C} K={self.K} iters={it + 1} ll={total / L:.4f}')
            self.models[pos] = dict(cells=cells, cidx={c: i for i, c in enumerate(cells)}, P=P, pi=pi)
        return self

    def posterior(self, pos, obs):
        """obs: list of (feats, class). Returns (K,) posterior over paradigms."""
        m = self.models[pos]
        logp = np.log(m['pi'] + 1e-12).copy()
        seen = {}
        for f, c in obs:
            if f in m['cidx'] and c is not None and c >= 0:
                seen.setdefault(f, []).append(c)
        for f, cs in seen.items():
            c = collections.Counter(cs).most_common(1)[0][0]
            logp += np.log(m['P'][:, m['cidx'][f], c] + 1e-12)
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def class_dist(self, pos, feats, obs, post=None):
        m = self.models[pos]
        if feats not in m['cidx']:
            return np.full(NC, 1.0 / NC)
        w = self.posterior(pos, obs) if post is None else post
        return w @ m['P'][:, m['cidx'][feats], :]

    def class_dist_forms(self, pos, feats, obs_forms):
        return self.class_dist(pos, feats, [(f, stress_class(fm)) for f, fm in obs_forms])
