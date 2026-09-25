"""Build per-lemma stress-class vectors and paradigm cluster labels along the global rank order."""
import pandas as pd, numpy as np, sys, collections, pickle
sys.path.insert(0, 'src')
from paradigm import stress_class, ParadigmModel, NC
tr = pd.read_csv('data/train.csv'); dv = pd.read_csv('data/dev.csv')
df = pd.concat([tr, dv]); df['cls'] = df.form_vvz.map(stress_class); df = df[df.cls >= 0]
L = pd.read_pickle('runs/exp/rng_paradigm/lemmas.pkl')
out = {}
for pos, d in df.groupby('pos'):
    g = d.groupby(['lemma_ru', 'feats']).cls.agg(lambda s: collections.Counter(s).most_common(1)[0][0])
    x = g.unstack(); cells = list(x.columns); A = x.fillna(-1).values.astype(int); lem = list(x.index)
    # exact-vector counting among fully observed lemmas
    full = (A >= 0).all(1)
    vecs = collections.Counter(tuple(r) for r in A[full])
    sizes = sorted(vecs.values(), reverse=True)
    print(pos, 'lemmas', len(lem), 'fully observed', full.sum(), 'distinct vectors', len(vecs), 'sizes top', sizes[:60], 'n singletons', sum(1 for s in sizes if s == 1))
    # cluster with ParadigmModel K=64, 3 seeds; label = seed0 label; confidence = agreement across seeds via co-clustering
    labs = []
    for s in range(3):
        pm = ParadigmModel(K=64, seed=s).fit(d)
        m = pm.models[pos]
        # assign each lemma to argmax paradigm posterior
        logp = np.log(m['prior'])[None, :].repeat(len(lem), 0)
        for j, c in enumerate(cells):
            cj = m['cidx'][c]
            ok = A[:, j] >= 0
            logp[ok] += np.log(m['P'][:, cj, :][:, A[ok, j]]).T
        labs.append(logp.argmax(1))
    labs = np.array(labs)
    # co-clustering agreement: for each lemma, fraction of other lemmas with same label under seed0 that also share label in seeds 1,2
    # simpler: label = seed-0; confidence = per-lemma mismatch count to its centroid
    lab0 = labs[0]
    cent = np.zeros((64, A.shape[1]), int)
    for k in range(64):
        m_ = A[lab0 == k]
        for j in range(A.shape[1]):
            v = m_[:, j]; v = v[v >= 0]
            cent[k, j] = np.bincount(v, minlength=NC).argmax() if len(v) else -1
    mism = ((A != cent[lab0]) & (A >= 0)).sum(1)
    used = np.bincount(lab0, minlength=64)
    print(pos, 'clusters used', (used > 0).sum(), 'sizes', sorted(used, reverse=True)[:70])
    print(pos, 'mismatch dist', np.bincount(mism)[:8], 'frac mism<=1', (mism <= 1).mean())
    out[pos] = dict(lemmas=lem, cells=cells, A=A, labels=labs, cent=cent, mism=mism)
pickle.dump(out, open('runs/exp/rng_paradigm/labels_k64.pkl', 'wb'))
