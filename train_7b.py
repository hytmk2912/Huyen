import os, time, glob, shutil
import numpy as np
import torch
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from model import GPTConfig, GPT

DATA_DIR = os.environ.get("DATA_DIR", "data/code_7b")
CKPT_DIR = os.environ.get("CKPT_DIR", "out-code-7b")
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "512"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "1"))
GRAD_ACCUM = 32
TARGET_HOURS = float(os.environ.get("TARGET_HOURS", "100"))
LOG_EVERY = 5
SAVE_EVERY_SEC = 900
KEEP = 2
DS_CFG = os.environ.get("DS_CFG", "ds_config_7b.json")

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

def rotate():
    tags = sorted(
        [p for p in glob.glob(os.path.join(CKPT_DIR, "step_*")) if os.path.isdir(p)],
        key=os.path.getmtime,
    )
    while len(tags) > KEEP:
        shutil.rmtree(tags.pop(0), ignore_errors=True)

os.makedirs(CKPT_DIR, exist_ok=True)
train_size = os.path.getsize(os.path.join(DATA_DIR, "train.bin")) // 2
tps = MICRO_BATCH * BLOCK_SIZE * GRAD_ACCUM
print(f"FP32 | tokens={train_size:,} | tokens/step={tps:,} | block={BLOCK_SIZE}")
print("GPU", torch.cuda.get_device_name(0), "VRAM", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2), "GB")

cfg = GPTConfig(
    block_size=BLOCK_SIZE, vocab_size=50304,
    n_layer=32, n_head=32, n_embd=4096,
    dropout=0.0, bias=False,
)
print("khoi tao GPT duoi ZeRO-3 (khong load full 7B len VRAM)")
with deepspeed.zero.Init(config_dict_or_path=DS_CFG, enabled=True):
    model = GPT(cfg)
print("GPT 7B-class (khong phai Llama) — ZeRO partition xong")

engine, _, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config=DS_CFG
)
print("deepspeed ready", flush=True)

it = seen = 0
try:
    path, st = engine.load_checkpoint(CKPT_DIR, tag="latest")
except Exception as e:
    path, st = None, {}
    print("khong load checkpoint:", e)
if path:
    it = int(st.get("it", 0))
    seen = int(st.get("tokens_seen", 0))
    print(f"RESUME iter={it:,} seen={seen:,}")
else:
    print("train tu dau")

t0 = last = time.time()
cal = False
while (time.time() - t0) / 3600 < TARGET_HOURS:
    x, y = get_batch("train")
    x, y = x.to(engine.device), y.to(engine.device)
    _, loss = engine(x, y)
    engine.backward(loss)
    engine.step()
    it += 1
    seen += tps
    if it == 2 and not cal:
        s = (time.time() - t0) / 2
        print(f"[TOC DO] {s:.1f}s/step {tps/max(s,0.01):.0f} tok/s", flush=True)
        cal = True
    if it % LOG_EVERY == 0:
        print(f"iter {it} {(time.time()-t0)/3600:.2f}h loss={loss.item():.4f} seen={seen:,}", flush=True)
    if time.time() - last >= SAVE_EVERY_SEC:
        engine.save_checkpoint(CKPT_DIR, tag=f"step_{it}", client_state={"it": it, "tokens_seen": seen})
        engine.save_checkpoint(CKPT_DIR, tag="latest", client_state={"it": it, "tokens_seen": seen})
        rotate()
        last = time.time()
        torch.cuda.empty_cache()

engine.save_checkpoint(CKPT_DIR, tag=f"step_{it}", client_state={"it": it, "tokens_seen": seen})
engine.save_checkpoint(CKPT_DIR, tag="latest", client_state={"it": it, "tokens_seen": seen})
rotate()
print(f"XONG iter={it:,} seen={seen:,}")
