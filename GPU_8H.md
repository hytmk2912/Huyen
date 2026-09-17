# Phiên GPU 8 giờ — model 7B (không dùng Llama)

## Quyết định kích thước

- Máy: RTX 5060 Ti 24GB VRAM + **28GB RAM** + 1600GB SSD, thuê 8 giờ.
- **Không train 30B từ đầu** trên cấu hình này. Optimizer AdamW của 30B cần hàng trăm GB state; với 28GB RAM máy sẽ gần như chỉ swap NVMe và gần như không học được trong 8 giờ.
- Repo đang 3B (`n_layer=32, n_head=20, n_embd=2560`). Nâng lên **~6.7B** kiểu GPT-3 6.7B:
  - `n_layer=32`, `n_head=32`, `n_embd=4096`, `block_size=1024`, `vocab_size=50304` (tiktoken gpt2).
- Kiến trúc **nanoGPT / GPT decoder**, inference bằng `sample_7b.py` + PyTorch. Không dùng Llama, llama.cpp, hay HuggingFace LlamaForCausalLM.

## Lưu checkpoint

- Chỉ giữ **2 checkpoint DeepSpeed mới nhất** trên đĩa GPU (đủ để resume).
- Convert checkpoint mới nhất → `model_fp32.pt` rồi đẩy Hugging Face Hub.
- Trước khi hết giờ (7.5h) rsync checkpoint + `model_fp32.pt` về VPS CPU (máy GPU xoá dữ liệu khi hết phiên).

## Bảo mật

**Không** đưa mật khẩu Gmail/Hugging Face vào script. Tạo token Hugging Face tại https://huggingface.co/settings/tokens (Write) rồi export `HF_TOKEN`.
Đổi mật khẩu Gmail ngay vì đã lộ trong chat.
