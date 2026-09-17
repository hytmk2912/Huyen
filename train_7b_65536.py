#!/usr/bin/env python3
import os, time, glob
import numpy as np
import torch
from model_7b_rope import GPT7B

DATA_DIR = os.environ.get("DATA_DIR", "data/reason_7b")
CKPT_DIR = os.environ.get("CKPT_DIR", "out-7b-65536")
BLOCK = int(os.environ.get("BLOCK_SIZE", "65536"))
TARGET_HOURS = float(os.environ.get("TARGET_HOURS", "12"))
LR = 1.5e-4
KEEP = 2

class CPUAdamW:
    def __init__(self, params, lr=1.5e-4, betas=(0.9, 0.95), eps=1e-8, wd=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.t = 0
        self.m = [torch.zeros(p.numel(), dtype=torch.float32) for p in self.params]
        self.v = [torch.zeros(p.numel(), dtype=torch.float32) for p in self.params]
    @torch.no_grad()
    def step(self):
        self.t += 1
        inv1, inv2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.detach().float().reshape(-1).cpu()
            if self.wd:
                g = g.add(p.detach().float().reshape(-1).cpu(), alpha=self.wd)
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            upd = (self.m[i] / inv1) / ((self.v[i] / inv2).sqrt().add_(self.eps))
            p.add_(upd.to(p.device, dtype=p.dtype).view_as(p), alpha=-self.lr)
            p.grad = None

def get_batch():
    path = os.path.join(DATA_DIR, "train.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    span = min(BLOCK, len(data) - 1)
    if span < 64:
        raise SystemExit("train.bin qua ngan")
    i = int(torch.randint(0, len(data) - span, (1,)).item())
    x = torch.from_numpy(data[i:i + span].astype(np.int64))[None]
    y = torch.from_numpy(data[i + 1:i + 1 + span].astype(np.int64))[None]
    return x, y, span

os.makedirs(CKPT_DIR, exist_ok=True)
device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
print("GPU", torch.cuda.get_device_name(0), "VRAM", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
model = GPT7B(n_layer=32, n_head=32, n_embd=4096, block_size=BLOCK)
n = sum(p.numel() for p in model.parameters())
print(f"7B RoPE params={n/1e9:.3f}B block={BLOCK}")
model.to(device=device, dtype=torch.float16)
model.train()
opt = CPUAdamW(model.parameters(), lr=LR)
it = seen = 0
latest = os.path.join(CKPT_DIR, "latest.pt")
if os.path.exists(latest):
    ck = torch.load(latest, map_location="cpu")
    model.load_state_dict(ck["model"])
    it, seen = ck.get("it", 0), ck.get("tokens_seen", 0)
    print("RESUME", it, seen)
t0 = last = time.time()
while (time.time() - t0) / 3600 < TARGET_HOURS:
    x, y, span = get_batch()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        _, loss = model(x.to(device), y.to(device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    it += 1
    seen += span
    print(f"iter {it} {(time.time()-t0)/3600:.2f}h loss={float(loss):.4f} seen={seen:,} vram={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
    if time.time() - last >= 900:
        cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload = {"model": cpu, "it": it, "tokens_seen": seen}
        torch.save(payload, os.path.join(CKPT_DIR, f"step_{it}.pt"))
        torch.save(payload, latest)
        files = sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")), key=os.path.getmtime)
        while len(files) > KEEP:
            os.remove(files.pop(0))
        last = time.time()
        print("saved", latest, flush=True)
print("XONG", it, seen)
