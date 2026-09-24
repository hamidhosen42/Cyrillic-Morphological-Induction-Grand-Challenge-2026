"""Explicit stress-paradigm model.

Each lemma has a hidden stress paradigm (~50-100 per POS). A form's stress class is
0 = initial syllable, else 1 + distance of the stressed vowel from the last vowel (capped at 4).
Given a lemma's observed cells we compute a posterior over paradigms and hence a distribution
over the stress class of an unseen cell. Used to re-rank beam candidates.
"""
import numpy as np, pandas as pd, collections

ACC = '́'
VOW = set('аеиоуыэюяёѣѫ')
NC = 5


def stress_idx(s):
    i = s.find(ACC)
    return sum(1 for c in s[:i] if c in VOW) - 1 if i >= 0 else -1


def n_vow(s):
    return sum(1 for c in s.replace(ACC, '') if c in VOW)


def stress_class(form):
    si = stress_idx(form); nv = n_vow(form)
    if si < 0:
        return -1
    if si == 0:
        return 0
    return int(min(1 + (nv - 1 - si), 4))


class ParadigmModel:
    def __init__(self, K=96, eps=0.08, iters=25, seed=0):
        self.K, self.eps, self.iters, self.seed = K, eps, iters, seed
        self.models = {}

    def fit(self, df):
        """df: rows with lemma_ru,pos,feats,form_vvz (all known forms)."""
        df = df.copy()
        df['cls'] = df.form_vvz.map(stress_class)
        df = df[df.cls >= 0]
        rng = np.random.default_rng(self.seed)
        for pos, d in df.groupby('pos'):
            # majority class per (lemma, feats) across dialects
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
            # per-paradigm per-cell class distribution (smoothed) and prior
            P = np.full((K, C, NC), 1.0)
            for k in range(K):
                m = A[asg == k]
                for j in range(C):
                    v = m[:, j]; v = v[v >= 0]
                    P[k, j] += np.bincount(v, minlength=NC) if len(v) else 0
            P /= P.sum(-1, keepdims=True)
            prior = np.bincount(asg, minlength=K) + 1.0; prior /= prior.sum()
            self.models[pos] = dict(cells=cells, cidx={c: i for i, c in enumerate(cells)}, P=P, prior=prior)
        return self

    def class_dist(self, pos, feats, observed):
        """observed: list of (feats, form). Returns (NC,) distribution over stress class for target feats."""
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


def rerank(cands, pos, feats, observed, pm, lam=1.0):
    """cands: list of (form, logp). Returns best form after adding lam*log P(stress class)."""
    dist = pm.class_dist(pos, feats, observed)
    best, bs = None, -1e18
    for form, lp in cands:
        c = stress_class(form)
        pc = dist[c] if c >= 0 else 1e-3
        s = lp + lam * np.log(pc + 1e-9)
        if s > bs:
            best, bs = form, s
    return best
