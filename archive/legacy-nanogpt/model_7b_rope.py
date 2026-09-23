import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rope_angles(T, hd, device, base=10000.0):
    half = hd // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)

class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.hd = d // n_head
        self.ln1 = nn.LayerNorm(d, bias=False)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = apply_rope(q.view(B, T, self.n_head, self.hd).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, T, self.n_head, self.hd).transpose(1, 2), cos, sin)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        h = self.ln2(x)
        x = x + self.out(F.gelu(self.fc(h)))
        return x

class GPT7B(nn.Module):
    def __init__(self, vocab=50304, n_layer=32, n_head=32, n_embd=4096, block_size=65536):
        super().__init__()
        self.block_size = block_size
        self.n_head = n_head
        self.hd = n_embd // n_head
        self.wte = nn.Embedding(vocab, n_embd)
        self.h = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.lm_head = nn.Linear(n_embd, vocab, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx)
        cos, sin = rope_angles(T, self.hd, idx.device)
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        for blk in self.h:
            x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
