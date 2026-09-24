"""Inference with the stress-conditioned local model + paradigm class decision.

For every row: class distribution from the paradigm model (joint per (lemma, feats) group),
decode the form for the top classes, optionally merge with external beam candidates
(Kaggle NN, list over models of list over rows of [(form, logp)...]).

  python local_infer.py --model runs/loc_dev --mode dev [--ext runs/kdev2/val_cands.pkl]
  python local_infer.py --model runs/loc_full --mode full --ext runs/test_cands_v1v3.pkl --out runs/sub_x.csv
"""
import argparse, os, sys, pickle, random, collections, math, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from paradigm import ParadigmModel, stress_class, NC
from model import Seq2Seq
from validate import load_splits, weighted_score, DATA

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, help='comma-separated model dirs (ensemble)')
p.add_argument('--mode', default='dev')
p.add_argument('--ext', default=None, help='external candidates pkl (dev: dict seg->models; full: models)')
p.add_argument('--out', default=None)
p.add_argument('--topc', type=int, default=3)
p.add_argument('--device', default='mps')
p.add_argument('--w_ext', type=float, default=1.0, help='weight of external NN candidates vs local model')
p.add_argument('--lam', type=float, default=2.0)
p.add_argument('--lam_new', type=float, default=1.0)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--beam', type=int, default=1)
p.add_argument('--no_local', action='store_true', help='choose among external candidates only (no local decoding)')
args = p.parse_args()
device = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')

src_code = open(os.path.join(os.path.dirname(__file__), 'local_train.py')).read()
exec(src_code[src_code.index("ACC = "):src_code.index("# ---------------- splits")], globals())  # defines Vocab, make_input, ...
model_dirs = args.model.split(',')
meta = pickle.load(open(f'{model_dirs[0]}/meta.pkl', 'rb')); vocab = meta['vocab']; targs = meta['args']
args.max_ctx = targs['max_ctx']

tr = pd.read_csv(f'{DATA}/train.csv'); dv = pd.read_csv(f'{DATA}/dev.csv')
wug = pd.read_csv(f'{DATA}/wug_seeds.csv'); te = pd.read_csv(f'{DATA}/test_features.csv'); wug['lemma_frequency'] = 60.0
if args.mode == 'dev':
    train_rows, known, val = load_splits()
    pm_fit = train_rows
else:
    known = pd.concat([tr, dv, wug]); pm_fit = pd.concat([tr, dv]); val = {'test': te}
known_map = {}
for l, f, dl, fm in zip(known.lemma_ru, known.feats, known.dialect, known.form_vvz):
    known_map.setdefault(l, []).append((f, dl, fm))
train_lemmas = set(pm_fit.lemma_ru)
pms = [ParadigmModel(K=300, seed=s).fit(pm_fit) for s in (0, 1, 2)]

models = []
for md in model_dirs:
    ta = pickle.load(open(f'{md}/meta.pkl', 'rb'))['args']
    m = Seq2Seq(len(vocab), ta['d_model'], 4, ta['layers'], ta['layers'], ta['ff'], 0.0, max_len=512).to(device)
    m.load_state_dict(torch.load(f'{md}/model_last.pt', map_location=device)); m.eval(); models.append(m)
model = models[0]


def pad(seqs):
    S = max(map(len, seqs)); s = torch.zeros(len(seqs), S, dtype=torch.long)
    for i, a in enumerate(seqs): s[i, :len(a)] = torch.tensor(a)
    return s


def decode_all(srcs):
    global model
    outs = []
    for m in models:
        model = m; outs.append(decode_scored(srcs))
    return outs


@torch.no_grad()
def decode_scored(srcs, bs=192):
    """greedy (or beam) decode + sequence log-prob"""
    order = np.argsort([len(s) for s in srcs]); res = [None] * len(srcs)
    for i in range(0, len(srcs), bs):
        idx = order[i:i + bs]
        s = pad([srcs[j] for j in idx]).to(device)
        if args.beam > 1:
            ys, sc = model.beam(s, vocab.stoi[BOS], vocab.stoi[EOS], beam=args.beam, return_all=True)
            for j, y, c in zip(idx, ys[:, 0].cpu().tolist(), sc[:, 0].cpu().tolist()):
                res[j] = (vocab.dec(y), c)
            continue
        mem, mask = model.encode(s)
        B = s.size(0)
        ys = torch.full((B, 1), vocab.stoi[BOS], dtype=torch.long, device=device)
        lp = torch.zeros(B, device=device); done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(40):
            logits = model.decode(ys, mem, mask)[:, -1]
            logp = torch.log_softmax(logits.float(), -1)
            nxt = logp.argmax(-1)
            lp += torch.where(done, torch.zeros_like(lp), logp.gather(1, nxt[:, None])[:, 0])
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1); done |= nxt == vocab.stoi[EOS]
            if done.all(): break
        for j, y, l in zip(idx, ys[:, 1:].cpu().tolist(), lp.cpu().tolist()):
            res[j] = (vocab.dec(y), l)
    return res


def run(df, ext_models=None, exclude_self=True):
    rows = list(df.itertuples())
    # class priors, joint per (lemma, feats)
    prior = []
    for r in rows:
        cands = [c for c in known_map.get(r.lemma_ru, []) if not (exclude_self and c[0] == r.feats and c[1] == r.dialect)]
        prior.append(sum(pm.class_dist(r.pos, r.feats, [(f, fm) for f, d, fm in cands]) for pm in pms) / len(pms))
    # external candidate class mass
    ext_mass = None
    if ext_models is not None:
        ext_mass = []
        for i in range(len(rows)):
            m = np.full(NC, 1e-9); tally = collections.defaultdict(float)
            for mm in ext_models:
                lst = mm[i]; lp = np.array([c[1] for c in lst]); pr = np.exp(lp - lp.max()); pr /= pr.sum()
                for (s, _), q in zip(lst, pr): tally[s] += q / len(ext_models)
            for s, q in tally.items():
                c = stress_class(s)
                if c >= 0: m[c] += q
            ext_mass.append((m, tally))
    # joint class decision per group
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows): groups[(r.lemma_ru, r.feats)].append(i)
    cls_choice = [None] * len(rows); cls_top = [None] * len(rows)
    for g, idx in groups.items():
        r0 = rows[idx[0]]
        lam = args.lam if r0.lemma_ru in train_lemmas else args.lam_new
        sc = lam * np.log(prior[idx[0]] + 1e-9)
        if ext_mass is not None:
            sc = sc + args.w_ext * sum(np.log(ext_mass[i][0]) for i in idx)
        order = np.argsort(-sc)
        for i in idx:
            cls_choice[i] = int(order[0]); cls_top[i] = [int(c) for c in order[:args.topc]]
    # decode the chosen class with the local model (and alternatives for tie-breaking with ext)
    srcs = []
    for i, r in enumerate(rows):
        cands = [c for c in known_map.get(r.lemma_ru, []) if not (exclude_self and c[0] == r.feats and c[1] == r.dialect)]
        srcs.append(vocab.enc(make_input(r.lemma_ru, r.feats, r.dialect, r.yat_flag, r.lemma_frequency, cls_choice[i], cands)))
    if args.no_local:
        dec = [('', -1e9)] * len(srcs)
    else:
        t0 = time.time(); decs = decode_all(srcs); print(f'decoded {len(srcs)} rows x {len(models)} models in {time.time() - t0:.0f}s', flush=True)
        # local vote: sum of probabilities over models
        dec = []
        for i in range(len(srcs)):
            v = collections.defaultdict(float)
            for d in decs: v[d[i][0]] += math.exp(d[i][1]) / len(decs)
            f, pmass = max(v.items(), key=lambda x: x[1]); dec.append((f, math.log(pmass + 1e-12)))
    out_local, out_merged = [], []
    for i, r in enumerate(rows):
        form, lp = dec[i]
        out_local.append(form)
        if ext_mass is None:
            out_merged.append(form); continue
        # merge: among external candidates of the chosen class, vote with local form
        c = cls_choice[i]; tally = ext_mass[i][1]
        votes = collections.defaultdict(float)
        for s, q in tally.items():
            if stress_class(s) == c: votes[s] += args.w_ext * q
        if not args.no_local:
            votes[form] += math.exp(lp) if stress_class(form) == c else 0.3 * math.exp(lp)
        if not votes:
            votes = dict(tally)
        out_merged.append(max(votes.items(), key=lambda x: x[1])[0])
    return out_local, out_merged


ext = pickle.load(open(args.ext, 'rb')) if args.ext else None
if args.mode == 'dev':
    preds_l, preds_m, golds = {}, {}, {}
    for seg, df in val.items():
        if args.limit: df = df.head(args.limit)
        e = ext[{'transfer': 'dev_transfer', 'complete': 'dev_complete', 'wug': 'wug', 'unseen': 'unseen_adj'}[seg]] if ext else None
        if e is not None and args.limit: e = [m[:args.limit] for m in e]
        pl, pm_ = run(df, e); preds_l[seg] = pl; preds_m[seg] = pm_; golds[seg] = list(df.form_vvz)
        print(seg, 'local EM', round(np.mean([a == b for a, b in zip(pl, golds[seg])]), 4), 'merged EM', round(np.mean([a == b for a, b in zip(pm_, golds[seg])]), 4), flush=True)
    print('LOCAL  weighted score:', weighted_score(preds_l, golds))
    if ext: print('MERGED weighted score:', weighted_score(preds_m, golds))
else:
    pl, pm_ = run(te, ext, exclude_self=False)
    pd.DataFrame({'id': te.id, 'form_vvz': pm_ if ext else pl}).to_csv(args.out, index=False)
    print('wrote', args.out)
