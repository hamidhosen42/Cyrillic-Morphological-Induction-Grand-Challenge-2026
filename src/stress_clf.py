"""Dedicated stress-class classifier for new lemmas (wug condition).

Input : [POS] [target feats tags] [dialect] lemma chars <sep> seed form chars (SEV seed cell, with accent)
Output: 3way stress class of the target cell (initial / stem-final / ending / other); ambiguous gold masks
        are trained with loss = -log sum_{c in mask} p_c.
Trained on train lemmas whose SEV seed cell is known, one example per non-seed row.

  python3 src/stress_clf.py --mode dev  --out runs/sclf_dev   # train without held-out lemmas, eval on holdout wug
  python3 src/stress_clf.py --mode full --out runs/sclf_full  # train on train+dev, predict test wug rows
"""
import argparse, collections, math, os, pickle, random, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'src', 'exp', 'stress-representation'))
from validate import load_splits, DATA
import eval_reps
from eval_reps import REPS, MaskParadigmModel
eval_reps.register_main()

p = argparse.ArgumentParser()
p.add_argument('--mode', default='dev')
p.add_argument('--out', default='runs/sclf_dev')
p.add_argument('--epochs', type=int, default=12)
p.add_argument('--d', type=int, default=192)
p.add_argument('--layers', type=int, default=3)
p.add_argument('--bs', type=int, default=256)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--device', default='mps')
args = p.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
REP = REPS['3way']; NCLS = REP.n
SEED = {'N': 'N;ACC;SG', 'V': 'V;PRS;3;SG'}

tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv')
if args.mode == 'dev':
    train_rows, known, val = load_splits()
    fit = train_rows
else:
    fit = pd.concat([tr, dv]); known = pd.concat([tr, dv, wug])
fit = fit[fit.pos.isin(['N', 'V'])]


def seed_of(df):
    s = df[(df.feats.map(lambda f: f in SEED.values())) & (df.dialect == 'SEV')]
    s = s[[SEED[p_] == f for p_, f in zip(s.pos, s.feats)]]
    return s.drop_duplicates('lemma_ru').set_index('lemma_ru').form_vvz.to_dict()


fit_seed = seed_of(fit)
# training examples: rows of lemmas with a seed, excluding the seed row itself
ex = fit[fit.lemma_ru.isin(fit_seed) & ~((fit.dialect == 'SEV') & (fit.feats == fit.pos.map(SEED)))].copy()
ex['mask'] = [REP.mask(f, l, p_, d) for f, l, p_, d in zip(ex.form_vvz, ex.lemma_ru, ex.pos, ex.dialect)]
ex = ex[ex['mask'] > 0]

chars = set(''.join(pd.concat([tr.lemma_ru, dv.lemma_ru, wug.lemma_ru, te.lemma_ru, tr.form_vvz, dv.form_vvz, wug.form_vvz])))
tags = set()
for f in pd.concat([tr.feats, te.feats]).unique():
    for q in f.split(';'): tags.add('[F:' + q + ']')
tags |= {'[D:SEV]', '[D:POM]', '[D:KAM]', '[SEP]', '[CLS]'}
itos = ['<pad>'] + sorted(tags) + sorted(chars); stoi = {t: i for i, t in enumerate(itos)}


def encode(lemma, feats, dialect, seed):
    toks = ['[CLS]'] + ['[F:' + q + ']' for q in feats.split(';')] + ['[D:' + dialect + ']'] + list(lemma) + ['[SEP]'] + list(seed)
    return [stoi[t] for t in toks if t in stoi]


class Clf(nn.Module):
    def __init__(self, V, d, L):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0); self.pos = nn.Embedding(128, d)
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, L); self.head = nn.Linear(d, NCLS)
    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))[None]
        h = self.enc(h, src_key_padding_mask=(x == 0))
        return self.head(h[:, 0])


def pad(seqs):
    S = min(128, max(map(len, seqs))); t = torch.zeros(len(seqs), S, dtype=torch.long)
    for i, s in enumerate(seqs): s = s[:S]; t[i, :len(s)] = torch.tensor(s)
    return t


X = [encode(l, f, d, fit_seed[l]) for l, f, d in zip(ex.lemma_ru, ex.feats, ex.dialect)]
M = torch.tensor([[(m >> c) & 1 for c in range(NCLS)] for m in ex['mask']], dtype=torch.float32)
print(f'train examples {len(X)}  lemmas {ex.lemma_ru.nunique()}  device {dev}', flush=True)
model = Clf(len(itos), args.d, args.layers).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps = args.epochs * math.ceil(len(X) / args.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
order_len = np.argsort([len(x) for x in X])
for ep in range(args.epochs):
    model.train(); t0 = time.time(); tot = 0; nb = 0
    batches = [order_len[i:i + args.bs] for i in range(0, len(X), args.bs)]; random.shuffle(batches)
    for b in batches:
        x = pad([X[i] for i in b]).to(dev); m = M[b].to(dev)
        logp = F.log_softmax(model(x), -1)
        loss = -torch.logsumexp(logp + torch.log(m + 1e-12), -1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); nb += 1
    print(f'epoch {ep} loss {tot / nb:.4f} {time.time() - t0:.0f}s', flush=True)
torch.save(model.state_dict(), os.path.join(ROOT, args.out, 'model.pt'))
pickle.dump(dict(itos=itos, args=vars(args)), open(os.path.join(ROOT, args.out, 'meta.pkl'), 'wb'))


@torch.no_grad()
def predict(rows_df, seeds):
    model.eval(); out = np.zeros((len(rows_df), NCLS))
    enc = [encode(l, f, d, seeds[l]) for l, f, d in zip(rows_df.lemma_ru, rows_df.feats, rows_df.dialect)]
    idx = np.argsort([len(e) for e in enc])
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]
        out[b] = F.softmax(model(pad([enc[j] for j in b]).to(dev)), -1).cpu().numpy()
    return out


if args.mode == 'dev':
    w = val['wug']; seeds = seed_of(known)
    w = w[w.lemma_ru.isin(seeds)]
    P = predict(w, seeds)
    pickle.dump(dict(index=w.index.values, P=P), open(os.path.join(ROOT, args.out, 'holdout_wug_probs.pkl'), 'wb'))
    # compare with the 3way paradigm prior on the same rows
    pms = [pickle.load(open(os.path.join(ROOT, f'runs/exp/stress-representation/pm_3way_K300_s{s}.pkl'), 'rb')) for s in range(3)]
    km = collections.defaultdict(list)
    for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz): km[l].append((f, d, fm))
    acc = collections.defaultdict(list); nonseed = []
    for k, r in enumerate(w.itertuples()):
        gm = REP.mask(r.form_vvz, r.lemma_ru, r.pos, r.dialect)
        if gm == 0: continue
        obs = [(f, REP.mask(fm, r.lemma_ru, r.pos, d)) for f, d, fm in km[r.lemma_ru]]
        pp = sum(pm.class_dist(r.pos, r.feats, obs) for pm in pms) / 3
        for name, dist in [('paradigm', pp), ('clf', P[k]), ('prod', pp * P[k]), ('prod_sqrt', pp * np.sqrt(P[k])), ('mix', 0.5 * pp + 0.5 * P[k])]:
            acc[name].append((gm >> int(np.argmax(dist))) & 1)
        nonseed.append(r.feats != SEED[r.pos])
    ns = np.array(nonseed)
    for k_, v in acc.items():
        v = np.array(v); print(f'{k_:10s} class acc all {v.mean():.4f}  non-seed {v[ns].mean():.4f}  n={len(v)}')
else:
    ws = seed_of(known)
    tw = te[te.lemma_ru.isin(ws)]
    P = predict(tw, ws)
    pickle.dump(dict(index=tw.index.values, P=P), open(os.path.join(ROOT, args.out, 'test_wug_probs.pkl'), 'wb'))
    print('wrote test probs', len(tw))
