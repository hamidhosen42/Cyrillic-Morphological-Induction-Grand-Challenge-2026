"""Local (MPS) training of a stress-class-conditioned char transformer.

Input:  [D:..][F:..]* [YAT][FQ] [SC:c] lemma <sep> (context forms)*
Output: inflected form (with combining acute).
The stress class c (0 = initial, k = 1+distance from last vowel) is given as an input token, so the
network only models the segmental part; the explicit paradigm model chooses c at inference.

  python local_train.py --mode dev  --out runs/loc_dev  --epochs 8
  python local_train.py --mode full --out runs/loc_full --epochs 8
"""
import argparse, os, random, time, json, math, pickle, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from paradigm import ParadigmModel, stress_class, NC
from model import Seq2Seq

p = argparse.ArgumentParser()
p.add_argument('--mode', default='dev', choices=['dev', 'full'])
p.add_argument('--epochs', type=int, default=8)
p.add_argument('--bs', type=int, default=128)
p.add_argument('--lr', type=float, default=7e-4)
p.add_argument('--d_model', type=int, default=256)
p.add_argument('--layers', type=int, default=3)
p.add_argument('--ff', type=int, default=768)
p.add_argument('--dropout', type=float, default=0.1)
p.add_argument('--max_ctx', type=int, default=6)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='runs/loc_dev')
p.add_argument('--eval_n', type=int, default=1500)
p.add_argument('--device', default='mps')
p.add_argument('--resume', default=None)
args = p.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv')
wug['lemma_frequency'] = 60.0
device = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
os.makedirs(args.out, exist_ok=True)

ACC = '́'
PAD, BOS, EOS, SEP, CTX = '<pad>', '<bos>', '<eos>', '<sep>', '<ctx>'
DPRI = {'SEV': 1, 'POM': 2, 'KAM': 3}


def freq_bucket(f):
    return f"[FQ{int(min(6, max(1, math.floor(math.log10(f)))))}]"


class Vocab:
    def __init__(self, chars, tags):
        self.itos = [PAD, BOS, EOS, SEP, CTX] + sorted(tags) + sorted(chars)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def enc(self, toks): return [self.stoi[t] for t in toks]
    def dec(self, ids):
        out = []
        for i in ids:
            t = self.itos[i]
            if t == EOS: break
            if t in (PAD, BOS): continue
            out.append(t)
        return ''.join(out)


def build_vocab(dfs):
    chars, tags = set(), set()
    for d in dfs:
        for s in d.lemma_ru: chars.update(s)
        if 'form_vvz' in d:
            for s in d.form_vvz.dropna(): chars.update(s)
        for f in d.feats.unique():
            for q in f.split(';'):
                tags.add('[F:' + q + ']'); tags.add('[CF:' + q + ']')
        for dl in d.dialect.unique():
            tags.add('[D:' + dl + ']'); tags.add('[CD:' + dl + ']')
    for y in (0, 1): tags.add(f'[YAT{y}]')
    for b in range(1, 7): tags.add(f'[FQ{b}]')
    for c in range(NC): tags.add(f'[SC{c}]')
    tags.add('[SAME]')
    return Vocab(chars, tags)


def order_context(cands, feats, dialect, k):
    parts = feats.split(';')
    tgt_num = 'SG' if 'SG' in parts else ('PL' if 'PL' in parts else '')
    tgt_ls = 'LONG' if 'LONG' in parts else ('SHORT' if 'SHORT' in parts else '')
    same = [c for c in cands if c[0] == feats]
    other = {}
    for c in cands:
        if c[0] == feats: continue
        pri = 0 if c[1] == dialect else DPRI[c[1]]
        if c[0] not in other or pri < other[c[0]][0]:
            other[c[0]] = (pri, c)
    def key(c):
        q = c[0].split(';')
        num = 'SG' if 'SG' in q else ('PL' if 'PL' in q else '')
        ls = 'LONG' if 'LONG' in q else ('SHORT' if 'SHORT' in q else '')
        return (0 if num == tgt_num else 1, 0 if ls == tgt_ls else 1, c[0])
    rest = sorted([v[1] for v in other.values()], key=key)
    return (sorted(same, key=lambda c: DPRI[c[1]]) + rest)[:k]


def make_input(lemma, feats, dialect, yat, freq, sc, cands):
    toks = ['[D:' + dialect + ']'] + ['[F:' + q + ']' for q in feats.split(';')]
    toks += [f'[YAT{int(yat)}]', freq_bucket(freq), f'[SC{sc}]'] + list(lemma) + [SEP]
    for f, d, form in order_context(cands, feats, dialect, args.max_ctx):
        toks += [CTX] + (['[SAME]'] if f == feats else []) + ['[CD:' + d + ']'] + ['[CF:' + q + ']' for q in f.split(';')] + list(form) + [SEP]
    return toks


def sample_train_context(rows, feats, dialect, pos, rng):
    r = rng.random()
    if pos == 'ADJ' and r < 0.15:
        return []
    if pos in ('N', 'V') and r < 0.30:
        seed_f = 'N;ACC;SG' if pos == 'N' else 'V;PRS;3;SG'
        seeds = [c for c in rows if c[0] == seed_f and c[1] == 'SEV']
        if seeds: return seeds
    if rng.random() < 0.45:
        rows = [c for c in rows if c[0] != feats]
    if not rows: return []
    if rng.random() < 0.5: return rows
    return rng.sample(rows, rng.randint(1, len(rows)))


# ---------------- splits ----------------
rng = random.Random(123)
if args.mode == 'dev':
    lem_adj = sorted(set(tr[tr.pos == 'ADJ'].lemma_ru))
    seedrows = tr[((tr.feats == 'N;ACC;SG') | (tr.feats == 'V;PRS;3;SG')) & (tr.dialect == 'SEV')]
    hold_wug = set(rng.sample(sorted(set(seedrows.lemma_ru)), 500))
    hold_adj = set(rng.sample(lem_adj, 250))
    train_rows = tr[~tr.lemma_ru.isin(hold_wug | hold_adj)]
    known = train_rows
    lf = set(zip(known.lemma_ru, known.feats))
    seen = np.array([(l, f) in lf for l, f in zip(dv.lemma_ru, dv.feats)])
    w = tr[tr.lemma_ru.isin(hold_wug)]
    wseed = w[((w.feats == 'N;ACC;SG') | (w.feats == 'V;PRS;3;SG')) & (w.dialect == 'SEV')]
    val_sets = {'dev_transfer': dv[seen], 'dev_complete': dv[~seen], 'wug': w.drop(wseed.index), 'unseen_adj': tr[tr.lemma_ru.isin(hold_adj)]}
    known = pd.concat([known, wseed])
    pm_fit = train_rows
else:
    train_rows = pd.concat([tr, dv])
    known = pd.concat([tr, dv, wug])
    val_sets = {'dev_sample': dv.sample(1500, random_state=0)}
    pm_fit = train_rows
train_rows = train_rows[train_rows.form_vvz.map(stress_class) >= 0]
known_map = {}
for l, f, dl, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, dl, fm))
vocab = build_vocab([tr, dv, wug, te])
pickle.dump({'vocab': vocab, 'args': vars(args)}, open(f'{args.out}/meta.pkl', 'wb'))
pm = ParadigmModel(K=300, seed=0).fit(pm_fit)
print('device', device, 'train rows', len(train_rows), {k: len(v) for k, v in val_sets.items()}, flush=True)


class TrainDS(Dataset):
    def __init__(self, df):
        self.rows = list(zip(df.lemma_ru, df.pos, df.feats, df.dialect, df.yat_flag, df.lemma_frequency, df.form_vvz))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        lemma, pos, feats, dialect, yat, freq, form = self.rows[i]
        r = random.Random(random.getrandbits(64))
        others = [c for c in known_map.get(lemma, []) if not (c[0] == feats and c[1] == dialect)]
        cands = sample_train_context(others, feats, dialect, pos, r)
        src = vocab.enc(make_input(lemma, feats, dialect, yat, freq, stress_class(form), cands))
        return src, vocab.enc([BOS] + list(form) + [EOS])


def pad(seqs):
    S = max(map(len, seqs)); s = torch.zeros(len(seqs), S, dtype=torch.long)
    for i, a in enumerate(seqs): s[i, :len(a)] = torch.tensor(a)
    return s


def collate(batch):
    srcs, tgts = zip(*batch); return pad(srcs), pad(tgts)


def eval_rows(df):
    """for each row: obs (feats, form) list excluding self, predicted class dist, gold class"""
    out = []
    for r in df.itertuples():
        cands = [c for c in known_map.get(r.lemma_ru, []) if not (c[0] == r.feats and c[1] == r.dialect)]
        dist = pm.class_dist(r.pos, r.feats, [(f, fm) for f, d, fm in cands])
        out.append((r, cands, dist, stress_class(r.form_vvz)))
    return out


@torch.no_grad()
def decode(model, srcs, bs=192):
    model.eval(); order = np.argsort([len(s) for s in srcs]); res = [None] * len(srcs)
    for i in range(0, len(srcs), bs):
        idx = order[i:i + bs]
        s = pad([srcs[j] for j in idx]).to(device)
        ys = model.greedy(s, vocab.stoi[BOS], vocab.stoi[EOS]).cpu().tolist()
        for j, y in zip(idx, ys): res[j] = vocab.dec(y)
    return res


def evaluate(model, ev):
    """oracle-class EM (segmental accuracy) and end-to-end EM with paradigm-predicted class"""
    srcs_or, srcs_pm = [], []
    for r, cands, dist, gc in ev:
        srcs_or.append(vocab.enc(make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, max(gc, 0), cands)))
        srcs_pm.append(vocab.enc(make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, int(dist.argmax()), cands)))
    po = decode(model, srcs_or); pp = decode(model, srcs_pm)
    golds = [r.form_vvz for r, _, _, _ in ev]
    return round(float(np.mean([a == b for a, b in zip(po, golds)])), 4), round(float(np.mean([a == b for a, b in zip(pp, golds)])), 4)


ds = TrainDS(train_rows)
approx = np.array([len(l) + 20 + sum(len(c[2]) + 6 for c in known_map.get(l, [])[:args.max_ctx]) for l in train_rows.lemma_ru])
class BucketSampler(torch.utils.data.Sampler):
    def __iter__(self):
        order = np.argsort(approx + np.random.randint(0, 40, size=len(ds)))
        batches = [order[i:i + args.bs].tolist() for i in range(0, len(ds) - args.bs + 1, args.bs)]
        random.shuffle(batches); return iter(batches)
    def __len__(self): return len(ds) // args.bs
dl = DataLoader(ds, batch_sampler=BucketSampler(), collate_fn=collate, num_workers=0)
model = Seq2Seq(len(vocab), args.d_model, 4, args.layers, args.layers, args.ff, args.dropout, max_len=512).to(device)
if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=device))
print('params %.1fM' % (sum(q.numel() for q in model.parameters()) / 1e6), 'steps/epoch', len(dl), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
total = args.epochs * len(dl); warm = 800
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
evs = {k: eval_rows(v.sample(min(args.eval_n, len(v)), random_state=0)) for k, v in val_sets.items()}
step = 0
for ep in range(args.epochs):
    model.train(); t0 = time.time(); tot = 0; n = 0
    for src, tgt in dl:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1; tot += loss.item(); n += 1
        if step % 200 == 0:
            print(f'ep {ep} step {step}/{len(dl)} loss {tot / n:.4f} {time.time() - t0:.0f}s', flush=True)
    res = {k: evaluate(model, ev) for k, ev in evs.items()}
    print(f'=== epoch {ep} loss {tot / n:.4f} time {time.time() - t0:.0f}s  (oracle-class EM, paradigm-class EM):', res, flush=True)
    torch.save(model.state_dict(), f'{args.out}/model_last.pt')
print('DONE', flush=True)
