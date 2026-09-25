"""Re-score a candidate (fam, seed, K, pattern, ordering) on a longer prefix with a bigger null and a
label-free pairwise class-vector agreement statistic. Usage: RNG_M=4000 python3 verify_hit.py py 1234 47 rb+rnd rank_global"""
import os, sys, numpy as np, pickle, pandas as pd
sys.path.insert(0, 'src/exp/rng_paradigm'); import rngsearch as R
fam, seed, K, pat, ordn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
oi = R.NAMES.index(ordn)
class A: smax = 1; ksweep = True; kmin = K; kmax = K; nord = 9
log = []; res = R.run_family(fam, np.array([seed]), A, log)
df = pd.DataFrame(res, columns=['kind', 'fam', 'seed', 'K_or_s', 'pat_or_o', 'ord', 'purity', 'z'])
row = df[(df.kind == 'par') & (df.pat_or_o == pat) & (df.ord == ordn)]
print(f'M={R.M}  purity/z for {fam} seed={seed} K={K} pat={pat} ord={ordn}:'); print(row.to_string())
# bigger null (3000 random sequences) for this (ordering, K)
lab, pos = R.LAB[:, oi], R.POS[:, oi]
mu, sd = R.purity_null(lab, pos, K, n=3000, seed=1); P = float(row.purity.iloc[0]) if len(row) else float('nan')
print(f'null (n=3000): mean {mu:.4f} sd {sd:.4f} -> z {(P - mu) / sd:.2f}')
