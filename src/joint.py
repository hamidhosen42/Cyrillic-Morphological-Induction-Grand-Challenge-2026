"""Joint per-lemma paradigm inference.

A lemma has one latent stress paradigm k. Evidence:
  - known forms (train rows / wug seed): hard observations of the class in those cells
  - the NN's candidate class mass for every *unlabelled* row of that lemma (test/holdout rows)
Posterior over k couples the unlabelled rows, so 3-4 weakly-informative rows sharpen each other.

  log P(k) = log prior[k] + Σ_obs log P[k, cell, c_obs] + Σ_rows log Σ_c P[k, cell, c] · mass(c)
  P(c | row) = Σ_k P(k|·) · P[k, cell_row, c]
"""
import collections
import numpy as np
from paradigm import ParadigmModel, stress_class, NC


class JointParadigm:
    def __init__(self, models, temp=1.0, obs_w=1.0, row_w=1.0):
        """models: list of fitted ParadigmModel (different seeds, averaged)."""
        self.pms = models
        self.temp = temp          # temperature on the NN mass evidence
        self.obs_w = obs_w        # weight of hard observations
        self.row_w = row_w        # weight of NN-row evidence

    def posterior(self, pm, pos, obs, rows):
        """obs: list of (feats, class); rows: list of (feats, mass[NC]). Returns (K,) posterior."""
        m = pm.models[pos]
        logp = np.log(m['prior']).copy()
        seen = {}
        for f, c in obs:
            if f in m['cidx'] and c >= 0:
                seen.setdefault(f, []).append(c)
        for f, cs in seen.items():
            c = collections.Counter(cs).most_common(1)[0][0]
            logp += self.obs_w * np.log(m['P'][:, m['cidx'][f], c])
        for f, mass in rows:
            if f not in m['cidx']:
                continue
            mm = np.power(np.maximum(mass, 1e-9), self.temp)
            mm = mm / mm.sum()
            logp += self.row_w * np.log((m['P'][:, m['cidx'][f], :] * mm[None, :]).sum(1) + 1e-12)
        w = np.exp(logp - logp.max())
        return w / w.sum()

    def class_dists(self, pos, obs, rows):
        """Returns list of (NC,) class distributions, one per row, averaged over paradigm models."""
        out = np.zeros((len(rows), NC))
        for pm in self.pms:
            m = pm.models[pos]
            w = self.posterior(pm, pos, obs, rows)
            for i, (f, _) in enumerate(rows):
                if f in m['cidx']:
                    out[i] += w @ m['P'][:, m['cidx'][f], :]
                else:
                    out[i] += 1.0 / NC
        out /= len(self.pms)
        return out / out.sum(1, keepdims=True)


def build_masses(tallies):
    """tallies: list over rows of {form: prob}. Returns list of (NC,) class-mass arrays."""
    out = []
    for t in tallies:
        m = np.full(NC, 1e-9)
        for s, q in t.items():
            c = stress_class(s)
            if c >= 0:
                m[c] += q
        out.append(m / m.sum())
    return out


def combine_tally(cands_per_model, i):
    tally = collections.defaultdict(float)
    for mdl in cands_per_model:
        lst = mdl[i]
        lp = np.array([c[1] for c in lst])
        pr = np.exp(lp - lp.max()); pr /= pr.sum()
        for (s, _), q in zip(lst, pr):
            tally[s] += q / len(cands_per_model)
    return tally


def decide(df, cands_per_model, known_map, jp, lam, lam_train=None, train_lemmas=(), group_by_cell=True):
    """Full decision: joint per-lemma class inference, then pick the best candidate of that class.
    Rows of the same (lemma, feats) are forced to share a class."""
    rows = list(df.itertuples())
    tallies = [combine_tally(cands_per_model, i) for i in range(len(rows))]
    masses = build_masses(tallies)
    by_lemma = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_lemma[r.lemma_ru].append(i)
    out = [None] * len(rows)
    for lemma, idx in by_lemma.items():
        pos = rows[idx[0]].pos
        obs = [(f, stress_class(fm)) for f, d, fm in known_map.get(lemma, [])]
        # merge rows of the same cell: average their class mass
        cells = collections.defaultdict(list)
        for i in idx:
            cells[rows[i].feats].append(i)
        cell_list = list(cells)
        cell_rows = [(f, np.mean([masses[i] for i in cells[f]], axis=0)) for f in cell_list]
        dists = jp.class_dists(pos, obs, cell_rows)
        lm = lam_train if (lam_train is not None and lemma in train_lemmas) else lam
        for f, d in zip(cell_list, dists):
            for i in cells[f]:
                sc = np.log(masses[i]) + lm * np.log(d + 1e-9)
                c = int(sc.argmax())
                best = max(((s, q) for s, q in tallies[i].items() if stress_class(s) == c),
                           key=lambda x: x[1], default=None)
                out[i] = best[0] if best else max(tallies[i].items(), key=lambda x: x[1])[0]
    return out
