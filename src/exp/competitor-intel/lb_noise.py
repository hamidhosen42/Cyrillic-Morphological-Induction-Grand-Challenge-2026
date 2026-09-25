"""Public-LB snapshot (2026-09-24, 30% of test) analysis: tiers, gap decomposition, sampling noise."""
import numpy as np, pandas as pd
lb = [("localAI",0.69891,29),("Sepp Mair",0.69834,11),("FOYSAL",0.69435,19),("GeanisML",0.69237,5),("Mario Nicolae",0.69214,13),
      ("Ethan Li",0.68847,6),("Michael Ibrahim",0.68077,7),("Hanbat/woominyo",0.67449,6),("OpenAI/nacl3084",0.67185,4),("keeaitec",0.67077,13),
      ("Md. Hamid Hosen",0.66871,17),("nasubiman",0.66560,14),("Matteo",0.66475,2),("sonic240612",0.66125,6),("Adnan Zaman",0.65826,12),
      ("Phantom F5",0.65382,20),("Abiram",0.63924,29),("Akalya",0.63128,13),("thisray",0.62929,18),("shash20124",0.60505,15)]
df = pd.DataFrame(lb, columns=["team","score","entries"]); print(df.to_string(index=False))
ours, top = 0.66871, 0.69891
print(f"gap to #1 {top-ours:.4f}; gap to #3 {0.69435-ours:.4f}; gap to #10 {0.67077-ours:.4f}")
seg_n = {'transfer':18000,'complete':15000,'wug':18260,'unseen':8740}; w = {'transfer':1,'complete':2,'wug':3,'unseen':4}
em = {'transfer':0.946,'complete':0.946,'wug':0.54,'unseen':0.383}      # our LB-probe estimates
rng = np.random.default_rng(0); ks = list(seg_n)
segs = np.concatenate([np.full(n,i) for i,n in enumerate(seg_n.values())])
W = np.array([w[k] for k in ks])[segs]; P = np.array([em[k] for k in ks])[segs]
hit = rng.random(len(W)) < P
sub = np.array([(W[idx]*hit[idx]).sum()/W[idx].sum() for idx in (rng.random(len(W))<0.30 for _ in range(1000))])
print(f"full-test weighted EM {(W*P).sum()/W.sum():.4f}; public-30% sampling sd of weighted EM {sub.std():.4f} -> score sd {0.9*sub.std():.4f}")
cer_wrong = 0.2538   # measured on holdout by headroom.py
for s in (ours, top, 0.69435):
    print(f"LB {s:.5f} -> implied weighted EM {(s-0.1+0.1*cer_wrong)/(0.9+0.1*cer_wrong):.4f}")
gap = top - ours
for k, share in [('transfer',.131),('complete',.218),('wug',.398),('unseen',.254)]:
    print(f"closing the gap purely via {k}: EM +{gap/(0.9*share):.3f}")
print(f"closing via wug+unseen equally: EM +{gap/(0.9*(.398+.254)):.3f} on both")
