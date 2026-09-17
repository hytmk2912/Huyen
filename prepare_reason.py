#!/usr/bin/env python3
import os, numpy as np, tiktoken
from datasets import load_dataset

OUT = os.environ.get("OUT_DIR", "data/reason_7b")
TARGET = int(os.environ.get("TARGET_TOKENS", "80000000"))
os.makedirs(OUT, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
BUF = []
n = 0

def add(text):
    global n
    if not text or n >= TARGET:
        return
    ids = enc.encode_ordinary(str(text)[:20000]) + [enc.eot_token]
    BUF.extend(ids)
    n += len(ids)

def dump(name):
    global BUF
    if not BUF:
        return
    arr = np.array(BUF, dtype=np.uint16)
    arr.tofile(os.path.join(OUT, name))
    print("wrote", name, len(arr))
    BUF = []

print("download mix reasoning EN/VI")
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, row in enumerate(ds):
        add(row.get("text", ""))
        if n >= TARGET * 0.55 or i > 200000:
            break
        if i % 2000 == 0:
            print("fineweb", i, "tok", n)
except Exception as e:
    print("fineweb skip", e)

for lang, name in (("en", "wikimedia/wikipedia"),):
    try:
        ds = load_dataset(name, "20231101.en", split="train", streaming=True)
        for i, row in enumerate(ds):
            add(row.get("text", ""))
            if n >= TARGET * 0.7 or i > 30000:
                break
    except Exception as e:
        print("wiki en skip", e)

try:
    ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
    for i, row in enumerate(ds):
        add(row.get("text", ""))
        if n >= TARGET * 0.82 or i > 20000:
            break
except Exception as e:
    print("wiki vi skip", e)

for hf, fields in (
    ("openai/gsm8k", ("question", "answer")),
    ("EleutherAI/arc", ("question",)),
):
    try:
        split = "train"
        kwargs = {"name": "main"} if "gsm8k" in hf else {"name": "ARC-Easy"}
        ds = load_dataset(hf, split=split, **kwargs)
        for row in ds:
            add(" ".join(str(row.get(f, "")) for f in fields))
            if n >= TARGET:
                break
    except Exception as e:
        print("task skip", hf, e)

try:
    ds = load_dataset("lighteval/MATH", split="train")
    for row in ds:
        add(str(row.get("problem", "")) + "\n" + str(row.get("solution", "")))
        if n >= TARGET:
            break
except Exception as e:
    print("math skip", e)

val = BUF[len(BUF)//20:] if len(BUF) > 10000 else BUF[:1000]
train = BUF[: max(1, len(BUF) - len(val))]
BUF = train
dump("train.bin")
BUF = val
dump("val.bin")
print("done tokens~", n)
