# gen_multihop_using_local_model — Hướng dẫn chạy

Pipeline gen multi-turn jailbreak conversation (6–9 turn) từ CSV NSFW prompt, dùng **local Gemma model** thông qua Hugging Face `transformers`.

Logic giống hệt `gen_multihop.py` (CSV parsing, JSON repair, async semaphore, atomic checkpoint, resume by id), chỉ thay backend từ OpenAI/Gemini API → `LocalGemmaService`.

---

## 1. Yêu cầu hệ thống

| | Khuyến nghị |
|---|---|
| Python | 3.10+ |
| GPU | 1× A100 80GB (hoặc 2× A100 40GB / 2× RTX 6000 Ada) cho `bfloat16` |
| RAM hệ thống | ≥ 64GB (cho phần shard offload) |
| Disk | ≥ 60GB trống cho weights (`google/gemma-4-26B-A4B-it`) |
| CUDA | 12.1+ (khớp với wheel `torch` đã build) |

> Nếu không đủ VRAM cho `bfloat16` thuần, xem mục **Quantization 4-bit** ở cuối.

---

## 2. Cài môi trường

### 2.1 Tạo venv & cài deps

```bash
cd /path/to/text_classifier/prepare_data/gemma4_ws

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell

pip install --upgrade pip wheel
```

### 2.2 Cài PyTorch khớp CUDA

Pin theo CUDA driver của máy (chạy `nvidia-smi` để xem version):

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only (chỉ để smoke test logic, không khả thi cho 26B)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2.3 Cài còn lại

```bash
pip install -r requirements.txt
```

`requirements.txt` đã có sẵn: `torch transformers huggingface_hub accelerate openai google-genai json-repair python-dotenv pandas pydantic Pillow pytest`.

### 2.4 Verify GPU thấy được

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| n_gpu:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Phải in ra `cuda: True` và list GPU. Nếu `False`, kiểm tra wheel `torch` có khớp CUDA không.

---

## 3. Hugging Face token (nếu model gated)

Gemma series yêu cầu accept license + login.

```bash
# 1. Mở https://huggingface.co/google/gemma-4-26B-A4B-it → Accept license
# 2. Tạo access token: https://huggingface.co/settings/tokens (read scope đủ)
# 3. Set env (KHÔNG commit vào git):
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Hoặc thêm vào `.env` ở thư mục cha (`prepare_data/.env`) — script tự load qua `dotenv`:

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 4. Tải model về local

### Cách A — để script tự tải (lười, chấp nhận tải lúc chạy lần đầu)

Lần chạy đầu tiên `LocalGemmaService` sẽ tự `snapshot_download` vào:

```
gemma4_ws/models/google__gemma-4-26B-A4B-it/
```

Lần sau gặp `config.json` ở đó → load thẳng, không tải lại.

### Cách B — tải trước thủ công (ổn định hơn, có thanh tiến trình rõ)

```bash
huggingface-cli login              # nhập HF_TOKEN

# Tải về thư mục mặc định mà script sẽ tìm
huggingface-cli download google/gemma-4-26B-A4B-it \
  --local-dir gemma4_ws/models/google__gemma-4-26B-A4B-it \
  --local-dir-use-symlinks False
```

### Cách C — tải vào path tuỳ ý

```bash
huggingface-cli download google/gemma-4-26B-A4B-it \
  --local-dir /data/models/gemma4 \
  --local-dir-use-symlinks False
```

Khi chạy script trỏ qua env:

```bash
export MULTIHOP_LOCAL_MODEL_DIR=/data/models/gemma4
```

### Verify weights tải xong

```bash
ls gemma4_ws/models/google__gemma-4-26B-A4B-it/ | head
# phải thấy: config.json, tokenizer.json, model-00001-of-XXXXX.safetensors, ...
```

---

## 5. Chuẩn bị input CSV

CSV bắt buộc cột `category`, kèm 1 trong các cột prompt: `prompt` / `rewrite` / `original_text` (resolve theo thứ tự đó).

Mặc định script đọc:

```
gemma4_ws/jailbreak_data/enhanced_pipeline_gemini_for_gen_conversation.csv
```

Override qua `--input-csv` hoặc `--prompt-column` nếu tên cột khác.

---

## 6. Chạy script

### 6.1 Smoke test 1 sample (kiểm load model OK)

```bash
cd gemma4_ws
python gen_multihop_using_local_model.py --mode test_sample
```

In JSON ra stdout + ghi vào `jailbreak_data/multihop_conversations_sample_test_local_gemma4.json`.

Chỉ in stdout, không ghi file:
```bash
python gen_multihop_using_local_model.py --mode test_sample --no-sample-file
```

### 6.2 Full pipeline

```bash
python gen_multihop_using_local_model.py --mode full_data
```

Output mặc định: `jailbreak_data/multihop_conversations_on_400_prompts_local_gemma4.json`.

Chạy 1 phần rồi resume:
```bash
# chạy 50 dòng đầu để test
python gen_multihop_using_local_model.py --mode full_data --max-prompts 50

# chạy phần còn lại (cùng output → tự resume các record conversation=null)
python gen_multihop_using_local_model.py --mode full_data
```

Override input/output:
```bash
python gen_multihop_using_local_model.py --mode full_data \
  --input-csv path/to/another.csv \
  --output path/to/out.json \
  --start-idx 100 --max-prompts 200
```

---

## 7. Pick CUDA GPU

Script truyền `device_map="auto"` cho `accelerate` → mặc định **shard model qua tất cả GPU `accelerate` thấy**. Có 3 cách kiểm soát:

### 7.1 Chọn 1 GPU duy nhất

`CUDA_VISIBLE_DEVICES` ẩn các GPU khác trước khi process Python khởi động:

```bash
# chỉ dùng GPU 0
CUDA_VISIBLE_DEVICES=0 python gen_multihop_using_local_model.py --mode test_sample

# chỉ dùng GPU 3
CUDA_VISIBLE_DEVICES=3 python gen_multihop_using_local_model.py --mode full_data
```

### 7.2 Shard qua nhiều GPU cụ thể

```bash
# dùng GPU 0 và 1 (split layer giữa 2 card)
CUDA_VISIBLE_DEVICES=0,1 python gen_multihop_using_local_model.py --mode full_data
```

`device_map="auto"` sẽ chia layer theo dung lượng VRAM từng card.

### 7.3 Force CPU (chỉ debug logic, không thực tế cho 26B)

```bash
MULTIHOP_LOCAL_DEVICE_MAP=cpu \
MULTIHOP_LOCAL_DTYPE=float32 \
python gen_multihop_using_local_model.py --mode test_sample
```

### 7.4 Custom device_map nâng cao

Nếu cần map chi tiết (vd. ép 1 layer ra CPU để giảm VRAM), set `MULTIHOP_LOCAL_DEVICE_MAP` thành tên scheme `accelerate` hỗ trợ:

| Giá trị | Ý nghĩa |
|---|---|
| `auto` | (default) accelerate tự balance |
| `balanced` | chia đều layer theo GPU |
| `balanced_low_0` | giữ GPU 0 ít hơn (chừa cho hoạt động khác) |
| `sequential` | nhồi đầy GPU 0 trước, rồi 1, rồi 2... |
| `cpu` | toàn bộ về CPU |

Ví dụ:
```bash
MULTIHOP_LOCAL_DEVICE_MAP=balanced_low_0 \
CUDA_VISIBLE_DEVICES=0,1,2 \
python gen_multihop_using_local_model.py --mode full_data
```

### 7.5 Theo dõi GPU lúc chạy

```bash
# terminal khác
watch -n 1 nvidia-smi
# hoặc
nvtop
```

---

## 8. Tuning generation & throughput

Mọi tham số set qua env trước khi chạy.

| Env | Default | Ý nghĩa |
|---|---|---|
| `MULTIHOP_LOCAL_MODEL` | `google/gemma-4-26B-A4B-it` | HF repo id |
| `MULTIHOP_LOCAL_MODEL_DIR` | `<script_dir>/models/<safe_name>` | Thư mục cache weights |
| `MULTIHOP_LOCAL_DEVICE_MAP` | `auto` | xem 7.4 |
| `MULTIHOP_LOCAL_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` / `auto` |
| `MULTIHOP_LOCAL_MAX_NEW_TOKENS` | `4096` | Hạn token output / 1 sample |
| `MULTIHOP_LOCAL_TEMPERATURE` | `0.7` | |
| `MULTIHOP_LOCAL_TOP_P` | `0.95` | |
| `MULTIHOP_LOCAL_DO_SAMPLE` | `1` | `0` để greedy |
| `MULTIHOP_LOCAL_CONCURRENT` | `1` | Số task song song (model là single-resident → tăng ít có lợi, có lock bảo vệ) |
| `MULTIHOP_CHECKPOINT_EVERY` | `1` | Ghi file mỗi N sample done |
| `MULTIHOP_PROGRESS_EVERY` | `10` | Log progress mỗi N sample done |
| `HF_TOKEN` | — | HF access token (gated repo) |

Ví dụ chạy với generation deterministic + giới hạn token nhỏ hơn:
```bash
MULTIHOP_LOCAL_DO_SAMPLE=0 \
MULTIHOP_LOCAL_MAX_NEW_TOKENS=2048 \
python gen_multihop_using_local_model.py --mode full_data
```

---

## 9. Quantization 4-bit (khi VRAM không đủ)

`LocalGemmaService` hiện không bật sẵn 4-bit. Nếu cần, sửa nhanh trong `llm_service.py` ngay phần `AutoModelForCausalLM.from_pretrained(...)`:

```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
self.hf_model = AutoModelForCausalLM.from_pretrained(
    str(path),
    device_map=device_map,
    quantization_config=bnb,
)
```

Cần thêm:
```bash
pip install bitsandbytes
```

VRAM yêu cầu giảm ~3.5×: 26B nf4 ≈ 16–18GB → fit 1× RTX 4090/A6000.

---

## 10. Troubleshooting

| Triệu chứng | Nguyên nhân thường gặp | Fix |
|---|---|---|
| `RuntimeError: LocalGemmaService requires torch, transformers, ...` | Thiếu deps | `pip install -r requirements.txt` + bước 2.2 |
| `OSError: ... is not a local folder and is not a valid model identifier` | Repo gated chưa accept license | Mở trang HF, accept; set `HF_TOKEN` |
| `CUDA out of memory` lúc load | VRAM thiếu | Dùng nhiều GPU (`CUDA_VISIBLE_DEVICES=0,1`) hoặc 4-bit (mục 9) |
| `CUDA out of memory` lúc generate | KV cache quá lớn | Giảm `MULTIHOP_LOCAL_MAX_NEW_TOKENS` |
| Output toàn `conversation: null` + `error: Response is empty` | Model refuse nội dung | Bình thường với prompt cực mạnh — kiểm tra `error` field, chỉnh sys_prompt nếu cần |
| Output `error: Parse failed ...` | Model trả non-JSON / thiếu array | Tăng `MULTIHOP_LOCAL_MAX_NEW_TOKENS`, hoặc giảm `temperature` |
| Tải HF chậm/fail nửa chừng | Mạng | Dùng `huggingface-cli download` (mục 4 cách B) — hỗ trợ resume |
| Script chạy nhưng GPU idle | `device_map` không thấy GPU | Verify mục 2.4; check `CUDA_VISIBLE_DEVICES` |

---

## 11. Output schema

Mỗi record trong file JSON output:

```json
{
  "id": 0,
  "category": "sexual",
  "prompt": "<original NSFW prompt>",
  "conversation": [
    {"turn_id": 1, "role": "user", "content": "..."},
    {"turn_id": 2, "role": "user", "content": "..."}
  ]
}
```

Nếu sample fail (API lỗi / parse fail / model refuse):

```json
{
  "id": 12,
  "category": "...",
  "prompt": "...",
  "conversation": null,
  "error": "<short error trace>"
}
```

Chạy lại cùng `--output` → tự skip các record đã có `conversation != null`, chỉ retry các record `null`.
