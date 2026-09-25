"""Does a lemma's stress class per cell become more predictable from its rank-neighbours' class vectors
(k nearest in global rank, same POS) than from (a) the global per-cell prior, (b) suffix-matched random lemmas?"""
import pandas as pd, numpy as np, pickle, sys
sys.path.insert(0, 'src'); from paradigm import NC
lab = pickle.load(open('runs/exp/rng_paradigm/labels_k64.pkl', 'rb'))
L = pd.read_pickle('runs/exp/rng_paradigm/lemmas_lab.pkl')
rank = dict(zip(L.lemma_ru, L['rank']))
rng = np.random.default_rng(0)
for pos in ['ADJ', 'N', 'V']:
    d = lab[pos]; A = d['A']; lem = np.array(d['lemmas']); r = np.array([rank[l] for l in lem]); C = A.shape[1]
    order = np.argsort(r); A = A[order]; lem = lem[order]; r = r[order]; n = len(lem)
    suf2 = np.array([l[-2:] for l in lem]); nv = np.array([sum(c in 'аеиоуыэюяёѣ' for c in l) for l in lem])
    prior = np.stack([np.bincount(A[:, j][A[:, j] >= 0], minlength=NC) for j in range(C)]).astype(float); prior /= prior.sum(1, keepdims=True)
    def score(pred_dist, mask):
        # accuracy of argmax and mean log-lik over observed cells
        acc = []; ll = []
        for j in range(C):
            ok = (A[:, j] >= 0) & mask
            p = pred_dist[ok, j]; y = A[ok, j]
            acc.append((p.argmax(1) == y).mean()); ll.append(np.log(p[np.arange(len(y)), y] + 1e-9).mean())
        return np.mean(acc), np.mean(ll)
    test = rng.random(n) < 0.3
    res = {}
    res['global prior'] = score(np.broadcast_to(prior[None], (n, C, NC)), test)
    for k in [4, 10, 30]:
        for kind in ['rank', 'suffix2+nv', 'random']:
            pd_ = np.zeros((n, C, NC)) + 0.5 * prior[None]
            for i in np.where(test)[0]:
                if kind == 'rank':
                    cand = np.concatenate([np.arange(max(0, i - k), i), np.arange(i + 1, min(n, i + 1 + k))])
                elif kind == 'suffix2+nv':
                    cand = np.where((suf2 == suf2[i]) & (nv == nv[i]))[0]; cand = cand[cand != i]
                    cand = rng.choice(cand, min(2 * k, len(cand)), replace=False) if len(cand) else cand
                else:
                    cand = rng.choice(n, 2 * k, replace=False); cand = cand[cand != i]
                cand = cand[~test[cand]]  # only train-side neighbours
                for j in range(C):
                    v = A[cand, j]; v = v[v >= 0]
                    if len(v): pd_[i, j] += np.bincount(v, minlength=NC)
            pd_ /= pd_.sum(2, keepdims=True)
            res[f'{kind} k={k}'] = score(pd_, test)
    print(pos, 'n test lemmas', test.sum())
    for kk, (a, l) in res.items(): print(f'   {kk:18s} acc {a:.4f} loglik {l:.4f}')
