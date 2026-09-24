# Context-aware char transformer for VVZ morphological inflection (Kaggle GPU script)
import os, random, time, json, math, pickle, sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

MODE = os.environ.get('MODE', 'dev')          # 'dev' -> validation splits; 'full' -> train on train+dev, predict test
EPOCHS = int(os.environ.get('EPOCHS', 10))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0').split(',')]
BS = 128; LR = 5e-4; D_MODEL = 384; LAYERS = 6; FF = 1536; DROPOUT = 0.15; BEAM = 8; MAX_SRC = 480
if os.environ.get('DEBUG'): BS, D_MODEL, LAYERS, FF = 64, 64, 2, 128
import glob
_c = glob.glob('/kaggle/input/**/train.csv', recursive=True)
DATA = os.path.dirname(_c[0]) if _c else os.path.join(os.path.dirname(__file__), '..', 'data')
print('DATA', DATA, flush=True)
OUT = '/kaggle/working' if os.path.exists('/kaggle/working') else os.path.join(os.path.dirname(__file__), '..', 'runs', 'local')
os.makedirs(OUT, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() and not os.environ.get('FORCE_CPU') else 'cpu'))
print('device', device, 'MODE', MODE, 'EPOCHS', EPOCHS, 'SEEDS', SEEDS, flush=True)

try:
    from rapidfuzz.distance import Levenshtein as Lev
    def edit(a, b): return Lev.distance(a, b)
except Exception:
    def edit(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

# ============================ data ============================
ACC = '́'
PAD, BOS, EOS, SEP, CTX = '<pad>', '<bos>', '<eos>', '<sep>', '<ctx>'
MAX_CTX = 12

def freq_bucket(f):
    return f"[FQ{int(min(6, max(1, math.floor(math.log10(f)))))}]"

class Vocab:
    def __init__(self, chars, tag_tokens):
        self.itos = [PAD, BOS, EOS, SEP, CTX] + sorted(tag_tokens) + sorted(chars)
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
            for p in f.split(';'):
                tags.add('[F:' + p + ']'); tags.add('[CF:' + p + ']')
        for dl in d.dialect.unique():
            tags.add('[D:' + dl + ']'); tags.add('[CD:' + dl + ']')
    for y in (0, 1): tags.add(f'[YAT{y}]')
    for b in range(1, 7): tags.add(f'[FQ{b}]')
    tags.add('[SAME]')
    return Vocab(chars, tags)

def target_tokens(lemma, feats, dialect, yat, freq):
    toks = ['[D:' + dialect + ']'] + ['[F:' + p + ']' for p in feats.split(';')]
    toks += [f'[YAT{int(yat)}]', freq_bucket(freq)]
    return toks + list(lemma) + [SEP]

def ctx_tokens(feats, dialect, form, same):
    toks = [CTX] + (['[SAME]'] if same else [])
    toks += ['[CD:' + dialect + ']'] + ['[CF:' + p + ']' for p in feats.split(';')]
    return toks + list(form) + [SEP]

DPRI = {'SEV': 1, 'POM': 2, 'KAM': 3}
def order_context(cands, feats, dialect, k=MAX_CTX):
    """Keep all same-feats forms (other dialects); for other feats keep one form per feats
    (prefer target dialect, then SEV/POM/KAM). Priority: same feats, then same number & long/short."""
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
        p = c[0].split(';')
        num = 'SG' if 'SG' in p else ('PL' if 'PL' in p else '')
        ls = 'LONG' if 'LONG' in p else ('SHORT' if 'SHORT' in p else '')
        return (0 if num == tgt_num else 1, 0 if ls == tgt_ls else 1, c[0])
    rest = sorted([v[1] for v in other.values()], key=key)
    return (sorted(same, key=lambda c: DPRI[c[1]]) + rest)[:k]

def make_input(lemma, feats, dialect, yat, freq, cands):
    toks = target_tokens(lemma, feats, dialect, yat, freq)
    for f, d, form in order_context(cands, feats, dialect):
        ct = ctx_tokens(f, d, form, f == feats)
        if len(toks) + len(ct) > MAX_SRC: break
        toks += ct
    return toks

def sample_train_context(rows, feats, dialect, pos, rng):
    r = rng.random()
    if pos == 'ADJ' and r < 0.12:
        return []
    if pos in ('N', 'V') and r < 0.30:
        seed_f = 'N;ACC;SG' if pos == 'N' else 'V;PRS;3;SG'
        seeds = [c for c in rows if c[0] == seed_f and c[1] == 'SEV']
        if seeds: return seeds
    if rng.random() < 0.45:
        rows = [c for c in rows if c[0] != feats]
    if not rows: return []
    if rng.random() < 0.5:
        return rows
    return rng.sample(rows, rng.randint(1, len(rows)))

# ============================ model ============================
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, enc_layers=4, dec_layers=4, ff=1024, dropout=0.15, max_len=640):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.tf = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff, dropout, batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
    def embed(self, x):
        p = torch.arange(x.size(1), device=x.device)[None]
        return self.drop(self.emb(x) * math.sqrt(self.d_model) + self.pos(p))
    def encode(self, src):
        mask = src == 0
        return self.tf.encoder(self.embed(src), src_key_padding_mask=mask), mask
    def decode(self, tgt_in, mem, mem_mask):
        T = tgt_in.size(1)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=tgt_in.device), 1)
        h = self.tf.decoder(self.embed(tgt_in), mem, tgt_mask=causal, tgt_key_padding_mask=(tgt_in == 0), memory_key_padding_mask=mem_mask)
        return self.out(h)
    def forward(self, src, tgt_in):
        mem, mask = self.encode(src)
        return self.decode(tgt_in, mem, mask)
    @torch.no_grad()
    def beam(self, src, bos, eos, beam=4, max_len=40):
        mem, mask = self.encode(src)
        B = src.size(0)
        mem = mem.repeat_interleave(beam, 0); mask = mask.repeat_interleave(beam, 0)
        ys = torch.full((B * beam, 1), bos, dtype=torch.long, device=src.device)
        scores = torch.zeros(B, beam, device=src.device); scores[:, 1:] = -1e9
        finished = torch.zeros(B * beam, dtype=torch.bool, device=src.device)
        for t in range(max_len):
            logp = F.log_softmax(self.decode(ys, mem, mask)[:, -1].float(), -1)
            V = logp.size(-1)
            logp = torch.where(finished[:, None], torch.full_like(logp, -1e9), logp)
            logp[finished, 0] = 0.0
            cand = (scores.view(-1, 1) + logp).view(B, beam * V)
            top, idx = cand.topk(beam, -1)
            bi = idx // V; ti = idx % V
            sel = ((torch.arange(B, device=src.device) * beam)[:, None] + bi).view(-1)
            ys = torch.cat([ys[sel], ti.view(-1, 1)], 1)
            finished = finished[sel] | (ti.view(-1) == eos) | (ti.view(-1) == 0)
            scores = top
            if finished.all(): break
        return ys.view(B, beam, -1)[:, :, 1:], scores

# ============================ setup ============================
tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv')
wug['lemma_frequency'] = 60.0
vocab = build_vocab([tr, dv, wug, te])
rng = random.Random(123)
if MODE == 'dev':
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
else:
    train_rows = pd.concat([tr, dv])
    known = pd.concat([tr, dv, wug])
    val_sets = {'dev_sample': dv.sample(2000, random_state=0)}
if os.environ.get('DEBUG'):
    train_rows = train_rows.head(3000); te = te.head(300); val_sets = {k: v.head(200) for k, v in val_sets.items()}
known_map = {}
for l, f, dl, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, dl, fm))
print('train rows', len(train_rows), {k: len(v) for k, v in val_sets.items()}, flush=True)

class TrainDS(Dataset):
    def __init__(self, df):
        self.rows = list(zip(df.lemma_ru, df.pos, df.feats, df.dialect, df.yat_flag, df.lemma_frequency, df.form_vvz))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        lemma, pos, feats, dialect, yat, freq, form = self.rows[i]
        r = random.Random(random.getrandbits(64))
        others = [c for c in known_map.get(lemma, []) if not (c[0] == feats and c[1] == dialect)]
        cands = sample_train_context(others, feats, dialect, pos, r)
        return vocab.enc(make_input(lemma, feats, dialect, yat, freq, cands)), vocab.enc([BOS] + list(form) + [EOS])

def eval_inputs(df, exclude_self=True):
    out = []
    for lemma, feats, dialect, yat, freq in zip(df.lemma_ru, df.feats, df.dialect, df.yat_flag, df.lemma_frequency):
        cands = [c for c in known_map.get(lemma, []) if not (exclude_self and c[0] == feats and c[1] == dialect)]
        out.append(vocab.enc(make_input(lemma, feats, dialect, yat, freq, cands)))
    return out

def pad(seqs):
    S = max(map(len, seqs)); s = torch.zeros(len(seqs), S, dtype=torch.long)
    for i, a in enumerate(seqs): s[i, :len(a)] = torch.tensor(a)
    return s

def collate(batch):
    srcs, tgts = zip(*batch)
    return pad(srcs), pad(tgts)

def predict(model, srcs, bs=512, beam=BEAM):
    """returns list of list of (string, logprob) per item (beam candidates)"""
    model.eval(); order = np.argsort([len(s) for s in srcs]); res = [None] * len(srcs)
    for i in range(0, len(srcs), bs):
        idx = order[i:i + bs]
        s = pad([srcs[j] for j in idx]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            ys, sc = model.beam(s, vocab.stoi[BOS], vocab.stoi[EOS], beam=beam)
        ys = ys.cpu().tolist(); sc = sc.cpu().tolist()
        for j, y, c in zip(idx, ys, sc):
            res[j] = [(vocab.dec(yy), cc) for yy, cc in zip(y, c)]
    return res

def score(preds, golds):
    em = np.mean([p == g for p, g in zip(preds, golds)])
    cer = np.mean([edit(p, g) / max(1, len(g)) for p, g in zip(preds, golds)])
    return round(float(em), 4), round(float(cer), 4)

def train_one(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds = TrainDS(train_rows)
    # length-bucketed batches (approximate length = lemma + all known forms) to cut padding
    approx = np.array([len(l) + 20 + sum(len(c[2]) + 6 for c in known_map.get(l, [])[:MAX_CTX]) for l in train_rows.lemma_ru])
    class BucketSampler(torch.utils.data.Sampler):
        def __init__(self): self.n = len(ds)
        def __iter__(self):
            noisy = approx + np.random.randint(0, 40, size=self.n)
            order = np.argsort(noisy)
            batches = [order[i:i + BS].tolist() for i in range(0, self.n - BS + 1, BS)]
            random.shuffle(batches)
            return iter(batches)
        def __len__(self): return self.n // BS
    dl = DataLoader(ds, batch_sampler=BucketSampler(), collate_fn=collate, num_workers=(2 if device.type == 'cuda' else 0), persistent_workers=device.type == 'cuda')
    model = Seq2Seq(len(vocab), D_MODEL, 4, LAYERS, LAYERS, FF, DROPOUT).to(device)
    print('params %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=0.01)
    total = EPOCHS * len(dl); warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    val_inputs = {k: (eval_inputs(v.head(3000)), list(v.head(3000).form_vvz)) for k, v in val_sets.items()}
    step = 0
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0; n = 0
        for src, tgt in dl:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(src, tgt[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            tot += loss.item(); n += 1
        if ep % 3 == 2 or ep == EPOCHS - 1:
            res = {}
            for k, (srcs, golds) in val_inputs.items():
                preds = [c[0][0] for c in predict(model, srcs, beam=1)]
                res[k] = score(preds, golds)
            print(f'seed {seed} epoch {ep} loss {tot / n:.4f} time {time.time() - t0:.0f}s', res, flush=True)
        else:
            print(f'seed {seed} epoch {ep} loss {tot / n:.4f} time {time.time() - t0:.0f}s', flush=True)
    torch.save(model.state_dict(), f'{OUT}/model_seed{seed}.pt')
    return model

def ensemble(cands_per_model):
    """cands_per_model: list over models of list over items of [(str, logp)...]. Vote by summed prob mass."""
    out = []
    n_items = len(cands_per_model[0])
    for i in range(n_items):
        tally = {}
        for m in cands_per_model:
            lst = m[i]
            lp = np.array([c[1] for c in lst]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
            for (s, _), p in zip(lst, pr):
                tally[s] = tally.get(s, 0) + p
        out.append(max(tally.items(), key=lambda x: x[1])[0])
    return out

all_val = {k: [] for k in val_sets}; all_test = []
test_inputs = eval_inputs(te, exclude_self=False)
for seed in SEEDS:
    t0 = time.time()
    model = train_one(seed)
    for k, v in val_sets.items():
        srcs = eval_inputs(v); golds = list(v.form_vvz)
        c = predict(model, srcs); all_val[k].append(c)
        print(f'seed {seed} FINAL beam {k}: single', score([x[0][0] for x in c], golds), 'ensemble-so-far', score(ensemble(all_val[k]), golds), flush=True)
    if MODE == 'full':
        c = predict(model, test_inputs); all_test.append(c)
        pickle.dump(all_test, open(f'{OUT}/test_cands.pkl', 'wb'))
        sub = pd.DataFrame({'id': te.id, 'form_vvz': ensemble(all_test)})
        sub.to_csv(f'{OUT}/submission.csv', index=False)
        print('wrote submission with', len(all_test), 'models', flush=True)
    print(f'seed {seed} done in {time.time() - t0:.0f}s', flush=True)
    pickle.dump(all_val, open(f'{OUT}/val_cands.pkl', 'wb'))
print('ALL DONE', flush=True)
