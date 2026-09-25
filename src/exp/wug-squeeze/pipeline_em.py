"""Wug-holdout EM of the current base pipeline decision (Kaggle cands, joint (lemma,feats) class decision) with a given prior."""
import sys, os, argparse; sys.path.insert(0, os.path.dirname(__file__))
from common import *
p = argparse.ArgumentParser(); p.add_argument('--prior', default=''); p.add_argument('--lams', default='1,1.5,2,3,4'); p.add_argument('--mix', type=float, default=1.0)
args = p.parse_args()
train_rows, known, wug_df, seeds = get_splits()
B = pickle.load(open(os.path.join(CACHE, 'baseline_wug.pkl'), 'rb')); pr_b, mass, gold = B['pr'], B['mass'], B['gold']
_, tallies = wug_masses(wug_df)
priors = {'paradigm': pr_b}
if args.prior:
    for pth in args.prior.split(','):
        d = pickle.load(open(pth, 'rb')); pw = d['pw'] if isinstance(d, dict) else d
        priors[os.path.basename(pth)] = pw
        priors['geo(' + os.path.basename(pth) + ',paradigm)'] = np.exp(0.5 * np.log(pw + 1e-9) + 0.5 * np.log(pr_b + 1e-9))
rows = list(wug_df.itertuples()); groups = collections.defaultdict(list)
for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
golds = list(wug_df.form_vvz)
for name, pr in priors.items():
    for lam in [float(x) for x in args.lams.split(',')]:
        cls = np.zeros(len(rows), dtype=int)
        for key, idx in groups.items():
            sc = sum(np.log(mass[i]) for i in idx) + lam * np.log(pr[idx[0]] + 1e-9)
            cls[idx] = int(sc.argmax())
        pred = []
        for i, t in enumerate(tallies):
            best = max(((s, q) for s, q in t.items() if stress_class(s) == cls[i]), key=lambda x: x[1], default=None)
            pred.append(best[0] if best else max(t.items(), key=lambda x: x[1])[0])
        em = np.mean([a == b for a, b in zip(pred, golds)]); ca = np.mean(cls == gold)
        print(f'{name:36s} lam={lam}: class acc {ca:.4f}  wug EM {em:.4f}  (n={len(rows)})', flush=True)
