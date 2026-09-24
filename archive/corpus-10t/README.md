# Phần corpus 10T token (đã cất)

Phần này nhắm tới quy mô pretrain (10.000 tỷ token), không liên quan tới fine-tune, nên đã được cất khỏi `local_ai/` ở mốc M1 (xem `TASKS.md`). Code không còn được import và test không còn chạy.

- `corpus.py`: code thu thập, tokenize, chia shard, tải lên Hugging Face (trước đây ở `local_ai/data/corpus.py`).
- `corpus_10t.json`, `smoke_real.json`: cấu hình (trước đây ở `configs/datasets/`).
- `test_corpus.py`: test đi kèm (trước đây nằm trong `tests/test_data_factory.py`).

Hàm quét khóa bí mật đã được tách sang `local_ai/data/secrets.py`, nên lệnh `python -m local_ai.data secret-scan` vẫn chạy như cũ.
