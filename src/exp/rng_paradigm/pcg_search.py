"""default_rng (PCG64) search via the real API: strided doubles (yat), integers array (paradigm), floor(random*K)."""
import numpy as np, pandas as pd, pickle, sys, time, argparse
sys.path.insert(0, 'src/exp/rng_paradigm')
import rngsearch as R
ap = argparse.ArgumentParser(); ap.add_argument('--s0', type=int, default=0); ap.add_argument('--s1', type=int, default=20000)
ap.add_argument('--chunk', type=int, default=500); ap.add_argument('--patmax', type=int, default=300); ap.add_argument('--out', default='runs/exp/rng_paradigm/pcg.pkl'); args = ap.parse_args()
M, T = R.M, R.T; t0 = time.time(); res = []; log = []
for a in range(args.s0, args.s1, args.chunk):
    seeds = np.arange(a, min(a + args.chunk, args.s1)); B = len(seeds)
    D = np.empty((B, T)); U32 = {}
    for i, s in enumerate(seeds): D[i] = np.random.default_rng(int(s)).random(T)
    for s in range(1, 11):
        for o in range(s):
            X = D[:, o::s][:, :M]
            if X.shape[1] < M: continue
            Z = (X @ R.YW) / R.YSD; zi = np.unravel_index(np.argmin(Z), Z.shape)
            res.append(('yat', 'pcg', int(seeds[zi[0]]), s, o, R.NAMES[zi[1]], -1, float(Z[zi])))
            for b, oo in zip(*np.where(Z < -7)): log.append(('yat', 'pcg', int(seeds[b]), s, o, R.NAMES[oo], -1, float(Z[b, oo])))
    for K in range(30, 65):
        streams = {}
        G = np.empty((B, M), np.int64)
        for i, s in enumerate(seeds): G[i] = np.random.default_rng(int(s)).integers(0, K, size=M)
        streams['integers'] = G
        # integers interleaved with random(): simulate via real API for pattern [int, rnd] and [rnd, int]
        for pat in (('int+rnd', 'rnd+int') if a < args.patmax else ()):
            Gm = np.empty((B, M), np.int64); Ym = np.empty((B, M))
            for i, s in enumerate(seeds):
                g = np.random.default_rng(int(s))
                if pat == 'int+rnd':
                    for j in range(M): Gm[i, j] = g.integers(0, K); Ym[i, j] = g.random()
                else:
                    for j in range(M): Ym[i, j] = g.random(); Gm[i, j] = g.integers(0, K)
            streams[pat] = (Gm, Ym)
        for s in (1, 2):
            for o in range(s): streams[f'floor_s{s}o{o}'] = np.floor(D[:, o::s][:, :M] * K).astype(np.int64)
        for name, st in streams.items():
            G, Y = (st if isinstance(st, tuple) else (st, None))
            for oi in range(4):
                P = R.purity(G, R.LAB[:, oi], R.POS[:, oi], K); Z = R.pz(P, oi, K); bi = int(np.argmax(Z))
                res.append(('par', 'pcg', int(seeds[bi]), K, name, R.NAMES[oi], float(P[bi]), float(Z[bi])))
                for b in np.where(Z > 8)[0]: log.append(('par', 'pcg', int(seeds[b]), K, name, R.NAMES[oi], float(P[b]), float(Z[b])))
                if Y is not None:
                    Zy = (Y @ R.YW[:, oi]) / R.YSD[oi]; byi = int(np.argmin(Zy))
                    res.append(('yat_pat', 'pcg', int(seeds[byi]), K, name, R.NAMES[oi], -1, float(Zy[byi])))
                    for b in np.where(Zy < -7)[0]: log.append(('yat_pat', 'pcg', int(seeds[b]), K, name, R.NAMES[oi], -1, float(Zy[b])))
    print(f'pcg seeds {a}.. t={time.time()-t0:.0f}s hits={len(log)}', flush=True)
pickle.dump(dict(res=res, log=log), open(args.out, 'wb'))
r = pd.DataFrame(res, columns=['kind', 'fam', 'seed', 'K_or_s', 'pat_or_o', 'ord', 'purity', 'z'])
for kind, g in r.groupby('kind'):
    print(kind); print((g.nsmallest(5, 'z') if kind.startswith('yat') else g.nlargest(5, 'z')).to_string())
print('hits', len(log), 'time', round(time.time() - t0))
