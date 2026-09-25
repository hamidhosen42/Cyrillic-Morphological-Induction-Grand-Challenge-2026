"""Teacher-forced rescoring of candidate pools with the Kaggle-trained seq2seq models (run locally).

Each Kaggle model only gave probability to candidates inside its own beam. This scores EVERY candidate of a
row's pool under a given model, so the ensemble can be combined properly.

  python3 src/rescore.py --kind v3 --weights runs/kdev2w/model_seed0.pt --mode dev  --pool runs/final/pool_dev.pkl  --out runs/final/sc_dev_kdev2.pkl
  python3 src/rescore.py --kind v1 --weights runs/kfull1/model_seed0.pt --mode full --pool runs/final/pool_full.pkl --out runs/final/sc_full_v1s0.pkl

pool pkl: dict key=(lemma, feats, dialect) -> list of candidate forms. out: dict key -> {form: logp}.
Input construction replicates kaggle/train_kaggle.py exactly for each kernel version:
  v1: MAX_CTX=6, old order_context (same feats, same dialect, same number), d=320 L=5 ff=1280 max_len=400
  v3: MAX_CTX=12, new order_context (all same-feats + one per other feats), MAX_SRC=480, d=384 L=6 ff=1536 max_len=640
"""
import argparse, collections, math, os, pickle, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from validate import load_splits, DATA

p = argparse.ArgumentParser()
p.add_argument('--kind', required=True, choices=['v1', 'v3'])
p.add_argument('--weights', required=True)
p.add_argument('--mode', default='full')
p.add_argument('--pool', required=True)
p.add_argument('--out', required=True)
p.add_argument('--device', default='mps')
p.add_argument('--bs', type=int, default=64, help='rows per batch')
args = p.parse_args()
dev = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')

PAD, BOS, EOS, SEP, CTX = '<pad>', '<bos>', '<eos>', '<sep>', '<ctx>'
CFG = {'v1': dict(d=320, L=5, ff=1280, max_len=400, max_ctx=6, max_src=10 ** 9),
       'v3': dict(d=384, L=6, ff=1536, max_len=640, max_ctx=12, max_src=480)}[args.kind]


def freq_bucket(f):
    return f"[FQ{int(min(6, max(1, math.floor(math.log10(f)))))}]"


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
    tags.add('[SAME]')
    itos = [PAD, BOS, EOS, SEP, CTX] + sorted(tags) + sorted(chars)
    return itos, {t: i for i, t in enumerate(itos)}


DPRI = {'SEV': 1, 'POM': 2, 'KAM': 3}


def order_v1(cands, feats, dialect, k):
    tgt_num = 'SG' if ';SG' in feats else ('PL' if ';PL' in feats else '')
    def key(c):
        f, d, _ = c
        num = 'SG' if ';SG' in f else ('PL' if ';PL' in f else '')
        return (0 if f == feats else 1, 0 if d == dialect else 1, 0 if num == tgt_num else 1, f, d)
    return sorted(cands, key=key)[:k]


def order_v3(cands, feats, dialect, k):
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


def make_input(lemma, feats, dialect, yat, freq, cands):
    toks = ['[D:' + dialect + ']'] + ['[F:' + q + ']' for q in feats.split(';')] + [f'[YAT{int(yat)}]', freq_bucket(freq)] + list(lemma) + [SEP]
    order = order_v1 if args.kind == 'v1' else order_v3
    for f, d, form in order(cands, feats, dialect, CFG['max_ctx']):
        ct = [CTX] + (['[SAME]'] if f == feats else []) + ['[CD:' + d + ']'] + ['[CF:' + q + ']' for q in f.split(';')] + list(form) + [SEP]
        if len(toks) + len(ct) > CFG['max_src']: break
        toks += ct
    return toks


class Seq2Seq(nn.Module):
    def __init__(self, V, d, nhead, L, ff, max_len):
        super().__init__()
        self.d_model = d
        self.emb = nn.Embedding(V, d, padding_idx=0); self.pos = nn.Embedding(max_len, d)
        self.tf = nn.Transformer(d, nhead, L, L, ff, 0.0, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, V); self.drop = nn.Dropout(0.0)
    def embed(self, x):
        return self.emb(x) * math.sqrt(self.d_model) + self.pos(torch.arange(x.size(1), device=x.device))[None]


tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv'); wug['lemma_frequency'] = 60.0
itos, stoi = build_vocab([tr, dv, wug, te])
if args.mode == 'dev':
    train_rows, known, val = load_splits()
    rows_df = pd.concat([val[s] for s in ['transfer', 'complete', 'wug', 'unseen']])
    excl_self = True
else:
    known = pd.concat([tr, dv, wug]); rows_df = te; excl_self = False
km = collections.defaultdict(list)
for l, f, d, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz): km[l].append((f, d, fm))

model = Seq2Seq(len(itos), CFG['d'], 4, CFG['L'], CFG['ff'], CFG['max_len']).to(dev)
sd = torch.load(os.path.join(ROOT, args.weights), map_location=dev)
model.load_state_dict(sd); model.eval()
pool = pickle.load(open(os.path.join(ROOT, args.pool), 'rb'))

items = []
for r in rows_df.drop_duplicates(['lemma_ru', 'feats', 'dialect']).itertuples():
    key = (r.lemma_ru, r.feats, r.dialect)
    cands = [c for c in pool.get(key, []) if c]
    if not cands: continue
    ctx = [c for c in km.get(r.lemma_ru, []) if not (excl_self and c[0] == r.feats and c[1] == r.dialect)]
    src = [stoi[t] for t in make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, ctx) if t in stoi]
    tgts = [[stoi[BOS]] + [stoi[ch] for ch in c if ch in stoi] + [stoi[EOS]] for c in cands]
    items.append((key, src, cands, tgts))
print(f'{args.kind} {args.weights}: {len(items)} rows, {sum(len(it[2]) for it in items)} candidates, device {dev}', flush=True)

out = {}
order = np.argsort([len(it[1]) for it in items])
t0 = time.time()
with torch.no_grad():
    for b in range(0, len(order), args.bs):
        batch = [items[i] for i in order[b:b + args.bs]]
        S = max(len(it[1]) for it in batch)
        src = torch.zeros(len(batch), S, dtype=torch.long)
        for k, it in enumerate(batch): src[k, :len(it[1])] = torch.tensor(it[1])
        src = src.to(dev); smask = src == 0
        mem = model.tf.encoder(model.embed(src), src_key_padding_mask=smask)
        rix, tg = [], []
        for k, it in enumerate(batch):
            for t in it[3]: rix.append(k); tg.append(t)
        T = max(len(t) for t in tg)
        tgt = torch.zeros(len(tg), T, dtype=torch.long)
        for j, t in enumerate(tg): tgt[j, :len(t)] = torch.tensor(t)
        tgt = tgt.to(dev); rix_t = torch.tensor(rix, device=dev)
        inp, gold = tgt[:, :-1], tgt[:, 1:]
        causal = torch.triu(torch.full((T - 1, T - 1), float('-inf'), device=dev), 1)
        h = model.tf.decoder(model.embed(inp), mem[rix_t], tgt_mask=causal, tgt_key_padding_mask=(inp == 0),
                             memory_key_padding_mask=smask[rix_t])
        lp = F.log_softmax(model.out(h).float(), -1).gather(2, gold[:, :, None])[:, :, 0]
        lp = (lp * (gold != 0)).sum(1).cpu().numpy()
        j = 0
        for k, it in enumerate(batch):
            out[it[0]] = {c: float(lp[j + m]) for m, c in enumerate(it[2])}
            j += len(it[2])
        if (b // args.bs) % 100 == 0:
            print(f'  {b}/{len(order)} rows {time.time() - t0:.0f}s', flush=True)
pickle.dump(out, open(os.path.join(ROOT, args.out), 'wb'))
print(f'wrote {args.out} ({len(out)} rows) in {time.time() - t0:.0f}s', flush=True)
