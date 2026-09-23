#!/usr/bin/env python3
"""3B nanoGPT FP32 — PyTorch only. Optimizer moments on CPU so 22GB VRAM can hold weights."""
import os, time, glob, math, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from model import GPTConfig, GPT

DATA_DIR = os.environ.get("DATA_DIR", "data/code_7b")
CKPT_DIR = os.environ.get("CKPT_DIR", "out-code-3b")
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "512"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "32"))
TARGET_HOURS = float(os.environ.get("TARGET_HOURS", "100"))
LR = float(os.environ.get("LR", "1.5e-4"))
LOG_EVERY = 5
SAVE_EVERY_SEC = 900
KEEP = 2

class CPUAdamW:
    def __init__(self, params, lr=1.5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [torch.zeros(p.numel(), dtype=torch.float32) for p in self.params]
        self.v = [torch.zeros(p.numel(), dtype=torch.float32) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1t = 1 - self.b1 ** self.t
        b2t = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.detach().float().reshape(-1).cpu()
            if self.wd != 0:
                g = g.add(p.detach().float().reshape(-1).cpu(), alpha=self.wd)
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            upd = (self.m[i] / b1t) / ((self.v[i] / b2t).sqrt().add_(self.eps))
            p.add_(upd.to(device=p.device, dtype=p.dtype).view_as(p), alpha=-self.lr)
            p.grad = None

def get_batch(split):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    if split != "train" and not os.path.exists(path):
        path = os.path.join(DATA_DIR, "train.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(data) - BLOCK_SIZE
    if n <= 0:
        raise SystemExit(f"file qua ngan: {path}")
    ix = torch.randint(n, (MICRO_BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x, y

def wrap_checkpoint(model):
    for blk in model.transformer.h:
        orig = blk.forward
        def _fwd(x, orig=orig):
            return checkpoint(orig, x, use_reentrant=False)
        blk.forward = _fwd

def rotate():
    files = sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")), key=os.path.getmtime)
    while len(files) > KEEP:
        os.remove(files.pop(0))

os.makedirs(CKPT_DIR, exist_ok=True)
assert torch.cuda.is_available(), "khong co CUDA"
device = "cuda"
print("GPU", torch.cuda.get_device_name(0), "VRAM", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
train_size = os.path.getsize(os.path.join(DATA_DIR, "train.bin")) // 2
tps = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
print(f"3B FP32 PT | tokens={train_size:,} | tokens/step={tps:,} | block={BLOCK_SIZE}")

cfg = GPTConfig(
    block_size=BLOCK_SIZE, vocab_size=50304,
    n_layer=32, n_head=20, n_embd=2560,
    dropout=0.0, bias=False,
)
model = GPT(cfg)
nparam = sum(p.numel() for p in model.parameters())
print(f"params={nparam/1e9:.3f}B  (khong phai Llama)")
model.to(device=device, dtype=torch.float32)
model.train()
wrap_checkpoint(model)
opt = CPUAdamW(model.parameters(), lr=LR)

it = seen = 0
latest = os.path.join(CKPT_DIR, "latest.pt")
if os.path.exists(latest):
    ck = torch.load(latest, map_location="cpu")
    model.load_state_dict(ck["model"])
    it = int(ck.get("it", 0))
    seen = int(ck.get("tokens_seen", 0))
    print(f"RESUME iter={it:,} seen={seen:,}")
else:
    print("train tu dau")

def save(tag):
    raw = model.state_dict()
    cpu = {k: v.detach().cpu() for k, v in raw.items()}
    path = os.path.join(CKPT_DIR, tag)
    torch.save({"model": cpu, "it": it, "tokens_seen": seen, "cfg": cfg.__dict__}, path)
    print("saved", path, flush=True)

t0 = last = time.time()
cal = False
accum = 0
while (time.time() - t0) / 3600 < TARGET_HOURS:
    x, y = get_batch("train")
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x, y)
    (loss / GRAD_ACCUM).backward()
    accum += 1
    if accum < GRAD_ACCUM:
        continue
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    accum = 0
    it += 1
    seen += tps
    if it == 2 and not cal:
        s = (time.time() - t0) / 2
        print(f"[TOC DO] {s:.1f}s/step {tps/max(s,0.01):.0f} tok/s", flush=True)
        cal = True
    if it % LOG_EVERY == 0:
        print(f"iter {it} {(time.time()-t0)/3600:.2f}h loss={loss.item():.4f} seen={seen:,} vram={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
    if time.time() - last >= SAVE_EVERY_SEC:
        save(f"step_{it}.pt")
        save("latest.pt")
        rotate()
        last = time.time()
        torch.cuda.empty_cache()

save(f"step_{it}.pt")
save("latest.pt")
print(f"XONG iter={it:,} seen={seen:,}")
