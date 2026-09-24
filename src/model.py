import math, torch, torch.nn as nn, torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, enc_layers=4, dec_layers=4,
                 ff=1024, dropout=0.15, max_len=400):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.tf = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff, dropout,
                                 batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)

    def embed(self, x):
        p = torch.arange(x.size(1), device=x.device)[None]
        return self.drop(self.emb(x) * math.sqrt(self.d_model) + self.pos(p))

    def encode(self, src):
        mask = src == 0
        mem = self.tf.encoder(self.embed(src), src_key_padding_mask=mask)
        return mem, mask

    def decode(self, tgt_in, mem, mem_mask):
        T = tgt_in.size(1)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=tgt_in.device), 1)
        h = self.tf.decoder(self.embed(tgt_in), mem, tgt_mask=causal,
                            tgt_key_padding_mask=(tgt_in == 0), memory_key_padding_mask=mem_mask)
        return self.out(h)

    def forward(self, src, tgt_in):
        mem, mask = self.encode(src)
        return self.decode(tgt_in, mem, mask)

    @torch.no_grad()
    def greedy(self, src, bos, eos, max_len=40):
        mem, mask = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), bos, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, mem, mask)[:, -1]
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == eos
            if done.all():
                break
        return ys[:, 1:]

    @torch.no_grad()
    def beam(self, src, bos, eos, beam=4, max_len=40, return_all=False):
        """Batched beam search. Returns best sequences (B, L) or list of (seq, logp) per item."""
        mem, mask = self.encode(src)
        B = src.size(0)
        mem = mem.repeat_interleave(beam, 0)
        mask = mask.repeat_interleave(beam, 0)
        ys = torch.full((B * beam, 1), bos, dtype=torch.long, device=src.device)
        scores = torch.zeros(B, beam, device=src.device)
        scores[:, 1:] = -1e9
        finished = torch.zeros(B * beam, dtype=torch.bool, device=src.device)
        for t in range(max_len):
            logp = F.log_softmax(self.decode(ys, mem, mask)[:, -1], -1)  # (B*beam, V)
            V = logp.size(-1)
            # finished beams: only allow pad with 0 cost
            logp = torch.where(finished[:, None], torch.full_like(logp, -1e9), logp)
            logp[finished, 0] = 0.0
            cand = scores.view(-1, 1) + logp  # (B*beam, V)
            cand = cand.view(B, beam * V)
            top, idx = cand.topk(beam, -1)
            bi = idx // V
            ti = idx % V
            base = (torch.arange(B, device=src.device) * beam)[:, None]
            sel = (base + bi).view(-1)
            ys = torch.cat([ys[sel], ti.view(-1, 1)], 1)
            finished = finished[sel] | (ti.view(-1) == eos) | (ti.view(-1) == 0)
            scores = top
            if finished.all():
                break
        ys = ys.view(B, beam, -1)[:, :, 1:]
        if return_all:
            return ys, scores
        return ys[:, 0]
