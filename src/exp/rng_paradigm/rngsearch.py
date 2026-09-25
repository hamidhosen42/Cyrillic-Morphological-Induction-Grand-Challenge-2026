"""RNG-sequence search: does yat_flag / paradigm label along lemma orderings match a seeded RNG stream?
Word-level simulation of Python random (MT19937 init_by_array) and numpy RandomState (MT19937 init_genrand).
"""
import numpy as np, pandas as pd, pickle, random, sys, time, argparse, os
M = int(os.environ.get('RNG_M', 1024))   # lemmas per ordering used for scoring
T = int(os.environ.get('RNG_T', 8 * M))  # raw 32-bit words per seed
L = pd.read_pickle('runs/exp/rng_paradigm/lemmas_lab.pkl')  # sorted by rank, has lab (-1 unknown), pos, yat_flag
POSI = {'ADJ': 0, 'V': 1, 'N': 2}

def orderings():
    out = {}
    def add(name, d):
        d = d.iloc[:M]
        out[name] = (d.yat_flag.values.astype(float), d.lab.values.astype(int), d.pos.map(POSI).values.astype(int))
    add('rank_global', L)
    for p in ['ADJ', 'V', 'N']: add(f'rank_{p}', L[L.pos == p])
    A = L.sort_values('lemma_ru'); add('alpha_global', A)
    for p in ['ADJ', 'V', 'N']: add(f'alpha_{p}', A[A.pos == p])
    add('rank_rev', L.iloc[::-1])
    return out
ORD = orderings()
NAMES = list(ORD)
YV = np.stack([ORD[n][0] for n in NAMES], 1)        # M x O
YW = np.where(YV > 0, 1.0 / YV.sum(0), -1.0 / (M - YV.sum(0)))  # mean-diff weights
LAB = np.stack([ORD[n][1] for n in NAMES], 1)       # M x O
POS = np.stack([ORD[n][2] for n in NAMES], 1)
C = LAB.max() + 1
N1 = YV.sum(0); N0 = M - N1
YSD = np.sqrt((1 / N1 + 1 / N0) / 12.0)              # null std of mean difference for uniforms

def words_py(seeds):
    W = np.empty((len(seeds), T), dtype=np.uint32)
    for i, s in enumerate(seeds):
        r = random.Random(int(s)); W[i] = [r.getrandbits(32) for _ in range(T)]
    return W

def words_np(seeds):
    W = np.empty((len(seeds), T), dtype=np.uint32)
    for i, s in enumerate(seeds):
        W[i] = np.random.RandomState(int(s)).randint(0, 2 ** 32, size=T, dtype=np.uint32)
    return W

def doubles(W):
    a = (W[:, :-1] >> 5).astype(np.float64); b = (W[:, 1:] >> 6).astype(np.float64)
    return (a * 67108864.0 + b) / 9007199254740992.0

def purity(G, lab, pos, K):
    """G: B x M ints in [0,K); lab: M (-1 unknown); pos: M. Returns B purity values (pos-specific buckets)."""
    ok = lab >= 0
    B = G.shape[0]
    bucket = pos[None, ok] * K + G[:, ok]                      # B x m
    flat = (np.arange(B)[:, None] * (3 * K) + bucket) * C + lab[None, ok]
    cnt = np.bincount(flat.ravel(), minlength=B * 3 * K * C).reshape(B, 3 * K, C)
    return cnt.max(2).sum(1) / ok.sum()

def purity_null(lab, pos, K, n=400, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.integers(0, K, size=(n, M))
    p = purity(G, lab, pos, K); return p.mean(), p.std() + 1e-9

NULL = {}
def pz(P, o, K):
    if (o, K) not in NULL: NULL[(o, K)] = purity_null(LAB[:, o], POS[:, o], K)
    mu, sd = NULL[(o, K)]; return (P - mu) / sd

def accept_stream(vals, acc, M_):
    """vals, acc: B x T. Return B x M_ of first M_ accepted values per row (rejection-sampling simulation)."""
    idx = np.argsort(~acc, axis=1, kind='stable')[:, :M_]
    return np.take_along_axis(vals, idx, 1)

def run_family(fam, seeds, args, log):
    W = words_py(seeds) if fam == 'py' else words_np(seeds)
    D = doubles(W)
    res = []
    # --- stage 1: yat mean-difference at fixed strides ---
    for s in range(1, args.smax + 1):
        for o in range(s):
            X = D[:, o::s][:, :M]
            if X.shape[1] < M: continue
            Z = (X @ YW) / YSD                                       # B x O
            zi = np.unravel_index(np.argmin(Z), Z.shape)
            res.append(('yat', fam, int(seeds[zi[0]]), s, o, NAMES[zi[1]], -1, float(Z[zi])))
            for b, oo in zip(*np.where(Z < -7)):
                log.append(('yat', fam, int(seeds[b]), s, o, NAMES[oo], -1, float(Z[b, oo])))
    # --- stage 2: paradigm purity ---
    if args.ksweep:
        for K in range(args.kmin, args.kmax + 1):
            k = int(K - 1).bit_length(); mask = (1 << k) - 1
            streams = {}
            if fam == 'py':
                vals = (W >> (32 - k)); acc = vals < K
            else:
                vals = (W & mask); acc = vals < K
            streams['rb'] = accept_stream(vals, acc, M).astype(np.int64)                 # pure randbelow/randint loop
            # randbelow interleaved with one random() per lemma: simulate pointer walk
            for pat in (['rb', 'rnd'], ['rnd', 'rb']):
                Gm = np.zeros((len(seeds), M), np.int64); Ym = np.zeros((len(seeds), M)); ptr = np.zeros(len(seeds), np.int64)
                rows = np.arange(len(seeds))
                for i in range(M):
                    for op in pat:
                        if op == 'rnd':
                            Ym[:, i] = D[rows, ptr]; ptr += 2
                        else:
                            todo = np.ones(len(seeds), bool)
                            while todo.any():
                                v = vals[rows[todo], ptr[todo]]; a = acc[rows[todo], ptr[todo]]
                                ptr[todo] += 1
                                sel = np.where(todo)[0][a]; Gm[sel, i] = v[a]
                                todo[np.where(todo)[0][a]] = False
                streams['+'.join(pat)] = (Gm, Ym)
            # floor(u*K) at strides 2,4 (random.choices / floor(random()*K))
            for s in (2, 4):
                for o in range(s):
                    X = D[:, o::s][:, :M]; streams[f'floor_s{s}o{o}'] = np.floor(X * K).astype(np.int64)
            for name, st in streams.items():
                G, Y = (st if isinstance(st, tuple) else (st, None))
                for oi in range(len(NAMES)):
                    if oi >= args.nord: break
                    P = purity(G, LAB[:, oi], POS[:, oi], K); Z = pz(P, oi, K)
                    bi = int(np.argmax(Z))
                    if Y is not None:
                        Zy = (Y @ YW[:, oi]) / YSD[oi]
                        byi = int(np.argmin(Zy)); res.append(('yat_pat', fam, int(seeds[byi]), K, name, NAMES[oi], -1, float(Zy[byi])))
                        for b in np.where(Zy < -7)[0]: log.append(('yat_pat', fam, int(seeds[b]), K, name, NAMES[oi], -1, float(Zy[b])))
                    res.append(('par', fam, int(seeds[bi]), K, name, NAMES[oi], float(P[bi]), float(Z[bi])))
                    for b in np.where(Z > 6)[0]: log.append(('par', fam, int(seeds[b]), K, name, NAMES[oi], float(P[b]), float(Z[b])))
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fam', default='py'); ap.add_argument('--s0', type=int, default=0); ap.add_argument('--s1', type=int, default=1000)
    ap.add_argument('--chunk', type=int, default=500); ap.add_argument('--smax', type=int, default=10)
    ap.add_argument('--ksweep', action='store_true'); ap.add_argument('--kmin', type=int, default=30); ap.add_argument('--kmax', type=int, default=64)
    ap.add_argument('--nord', type=int, default=4); ap.add_argument('--out', default='runs/exp/rng_paradigm/out.pkl')
    ap.add_argument('--special', action='store_true')
    args = ap.parse_args()
    t0 = time.time(); allres = []; log = []
    if args.special:
        seeds = [42, 0, 1, 2, 3, 7, 10, 11, 13, 17, 21, 23, 31, 37, 41, 43, 69, 77, 99, 100, 101, 111, 123, 321, 420, 666, 777, 999, 1000, 1001, 1234, 1337, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 3407, 4242, 5555, 8888, 9999, 12345, 31337, 54321, 65536, 123456, 424242, 654321, 999999, 1000000, 1234567, 3141592, 314159, 271828, 8675309, 12345678, 123456789, 987654321, 20240101, 20250101, 20260101, 20260901, 20260924, 2026001, 1618033, 4294967295, 2147483647, 1000000007, 0xDEADBEEF, 0xCAFEBABE, 0xBADF00D, 0xABCDEF, 2 ** 31, 2 ** 32, 2 ** 32 - 1, 2 ** 40, 2 ** 63 - 1]
        if args.fam == 'np': seeds = [s for s in seeds if s < 2 ** 32]
        seeds = np.array(seeds)
        allres += run_family(args.fam, seeds, args, log)
    else:
        for a in range(args.s0, args.s1, args.chunk):
            seeds = np.arange(a, min(a + args.chunk, args.s1))
            allres += run_family(args.fam, seeds, args, log)
            if (a // args.chunk) % 10 == 0: print(f'{args.fam} seeds {a}.. t={time.time()-t0:.0f}s hits={len(log)}', flush=True)
    pickle.dump(dict(res=allres, log=log, args=vars(args)), open(args.out, 'wb'))
    # summary
    r = pd.DataFrame(allres, columns=['kind', 'fam', 'seed', 'K_or_s', 'pat_or_o', 'ord', 'purity', 'z'])
    for kind, g in r.groupby('kind'):
        if kind.startswith('yat'): print(kind, 'most negative z:'); print(g.nsmallest(5, 'z').to_string())
        else: print(kind, 'largest z:'); print(g.nlargest(5, 'z').to_string())
    print('n logged hits', len(log), 'time', round(time.time() - t0), 's')
