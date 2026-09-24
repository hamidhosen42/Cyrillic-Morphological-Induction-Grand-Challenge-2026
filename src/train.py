import argparse, os, random, time, json, math, pickle
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rapidfuzz.distance import Levenshtein as Lev
from data import *
from model import Seq2Seq

p = argparse.ArgumentParser()
p.add_argument('--mode', default='dev', choices=['dev', 'full'])
p.add_argument('--epochs', type=int, default=12)
p.add_argument('--bs', type=int, default=192)
p.add_argument('--lr', type=float, default=6e-4)
p.add_argument('--d_model', type=int, default=256)
p.add_argument('--layers', type=int, default=4)
p.add_argument('--ff', type=int, default=1024)
p.add_argument('--dropout', type=float, default=0.15)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='ckpt')
p.add_argument('--eval_n', type=int, default=3000)
p.add_argument('--workers', type=int, default=0)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv')
wug['lemma_frequency'] = 60.0
vocab = build_vocab([tr, dv, wug, te])
os.makedirs(args.out, exist_ok=True)
pickle.dump(vocab, open(f'{args.out}/vocab.pkl', 'wb'))
dev_ = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device(dev_)
print('vocab', len(vocab), 'device', device)

# ---------------- splits ----------------
rng = random.Random(123)
if args.mode == 'dev':
    lem_nv = sorted(set(tr[tr.pos.isin(['N', 'V'])].lemma_ru))
    lem_adj = sorted(set(tr[tr.pos == 'ADJ'].lemma_ru))
    # wug-like holdout: N/V lemmas that have the SEV seed cell in train
    seedrows = tr[((tr.feats == 'N;ACC;SG') | (tr.feats == 'V;PRS;3;SG')) & (tr.dialect == 'SEV')]
    seed_lemmas = sorted(set(seedrows.lemma_ru))
    hold_wug = set(rng.sample(seed_lemmas, 500))
    hold_adj = set(rng.sample(lem_adj, 250))
    hold = hold_wug | hold_adj
    train_rows = tr[~tr.lemma_ru.isin(hold)]
    known = train_rows  # context source
    val_sets = {}
    d = dv.copy()
    lf = set(zip(known.lemma_ru, known.feats))
    seen = np.array([(l, f) in lf for l, f in zip(d.lemma_ru, d.feats)])
    val_sets['dev_transfer'] = d[seen]
    val_sets['dev_complete'] = d[~seen]
    w = tr[tr.lemma_ru.isin(hold_wug)]
    wseed = w[((w.feats == 'N;ACC;SG') | (w.feats == 'V;PRS;3;SG')) & (w.dialect == 'SEV')]
    val_sets['wug'] = w.drop(wseed.index)
    val_sets['unseen_adj'] = tr[tr.lemma_ru.isin(hold_adj)]
    known = pd.concat([known, wseed])
else:
    train_rows = pd.concat([tr, dv])
    known = pd.concat([tr, dv, wug])
    val_sets = {'dev_sample': dv.sample(2000, random_state=0)}

# known forms per lemma
known_map = {}
for l, f, dl, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, dl, fm))
print('train rows', len(train_rows), {k: len(v) for k, v in val_sets.items()})


class TrainDS(Dataset):
    def __init__(self, df):
        self.rows = list(zip(df.lemma_ru, df.pos, df.feats, df.dialect, df.yat_flag, df.lemma_frequency, df.form_vvz))
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        lemma, pos, feats, dialect, yat, freq, form = self.rows[i]
        r = random.Random(hash((i, random.random())))
        others = [c for c in known_map.get(lemma, []) if not (c[0] == feats and c[1] == dialect)]
        cands = sample_train_context(others, feats, dialect, pos, r)
        src = vocab.enc(make_input(lemma, feats, dialect, yat, freq, cands))
        tgt = vocab.enc([BOS] + list(form) + [EOS])
        return src, tgt


def eval_inputs(df, exclude_self=True):
    out = []
    for lemma, feats, dialect, yat, freq in zip(df.lemma_ru, df.feats, df.dialect, df.yat_flag, df.lemma_frequency):
        cands = [c for c in known_map.get(lemma, []) if not (exclude_self and c[0] == feats and c[1] == dialect)]
        out.append(vocab.enc(make_input(lemma, feats, dialect, yat, freq, cands)))
    return out


def collate(batch):
    srcs, tgts = zip(*batch)
    S = max(map(len, srcs)); T = max(map(len, tgts))
    s = torch.zeros(len(batch), S, dtype=torch.long); t = torch.zeros(len(batch), T, dtype=torch.long)
    for i, (a, b) in enumerate(zip(srcs, tgts)):
        s[i, :len(a)] = torch.tensor(a); t[i, :len(b)] = torch.tensor(b)
    return s, t


def pad_srcs(srcs):
    S = max(map(len, srcs))
    s = torch.zeros(len(srcs), S, dtype=torch.long)
    for i, a in enumerate(srcs):
        s[i, :len(a)] = torch.tensor(a)
    return s


def predict(model, srcs, bs=256):
    model.eval(); preds = []
    order = np.argsort([len(s) for s in srcs])
    res = [None] * len(srcs)
    for i in range(0, len(srcs), bs):
        idx = order[i:i + bs]
        s = pad_srcs([srcs[j] for j in idx]).to(device)
        ys = model.greedy(s, vocab.stoi[BOS], vocab.stoi[EOS]).cpu().tolist()
        for j, y in zip(idx, ys):
            res[j] = vocab.dec(y)
    return res


def score(preds, golds):
    em = np.mean([p == g for p, g in zip(preds, golds)])
    cer = np.mean([Lev.distance(p, g) / max(1, len(g)) for p, g in zip(preds, golds)])
    return em, cer


ds = TrainDS(train_rows)
dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=args.workers,
                persistent_workers=args.workers > 0, drop_last=True)
model = Seq2Seq(len(vocab), args.d_model, 4, args.layers, args.layers, args.ff, args.dropout).to(device)
print('params', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
total = args.epochs * len(dl); warm = 1000
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

val_inputs = {k: (eval_inputs(v.head(args.eval_n)), list(v.head(args.eval_n).form_vvz)) for k, v in val_sets.items()}
step = 0; best = -1
for ep in range(args.epochs):
    model.train(); t0 = time.time(); tot = 0; n = 0
    for src, tgt in dl:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        tot += loss.item(); n += 1
        if step % 200 == 0:
            print(f'ep {ep} step {step} loss {tot / n:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s', flush=True)
    res = {}
    for k, (srcs, golds) in val_inputs.items():
        preds = predict(model, srcs)
        em, cer = score(preds, golds); res[k] = (round(em, 4), round(cer, 4))
    print(f'=== epoch {ep} loss {tot / n:.4f} time {time.time() - t0:.0f}s', res, flush=True)
    torch.save(model.state_dict(), f'{args.out}/model_ep{ep}.pt')
    torch.save(model.state_dict(), f'{args.out}/model_last.pt')
json.dump(vars(args), open(f'{args.out}/args.json', 'w'))
