"""Inference nanoGPT 7B-class — khong dung Llama / llama.cpp."""
import torch
import tiktoken
from model import GPTConfig, GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
config = GPTConfig(
    block_size=1024, vocab_size=50304,
    n_layer=32, n_head=32, n_embd=4096,
    dropout=0.0, bias=False,
)
model = GPT(config)
state = torch.load("model_fp32.pt", map_location="cpu")
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.to(device)
if device == "cuda":
    model.half()
model.eval()
enc = tiktoken.get_encoding("gpt2")
PROMPTS = {
    "Code": "def binary_search(arr, target):",
    "Reasoning": "Question: A shop has 15 apples, sells 6, then gets 20 more. How many now?\nAnswer:",
    "Vietnamese": "Ha Noi la thu do cua",
    "Bugfix": "# Bug: crashes on empty list\ndef get_first(items):\n    return items[0]\n# Fixed:\ndef get_first(items):",
}
for label, prompt in PROMPTS.items():
    ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=50)
    print(f"\n=== {label} ===")
    print(enc.decode(out[0].tolist()))
