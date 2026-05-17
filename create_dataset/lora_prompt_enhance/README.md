# Qwen2.5-7B LoRA for Prompt Enhancement

Train LoRA to learn style from your paired dataset:
- Input: `original_text` (raw prompt)
- Output: `rewritten_text` (enhanced prompt)
- Control fields used in prompt formatting:
  - `category`

This version is `category-only` (no `toxicity_level` control in train or inference).

## Current Project Structure

From workspace root (`/media02/ltnghia24`):

```text
create_dataset/
├── data/
│   └── nsfw_prompts.csv
└── lora_prompt_enhance/
    ├── prepare_dataset.py
    ├── train_lora.py
    ├── inference_enhance.py
    ├── run_training_job.sh     ← SLURM job script
    ├── requirements.txt
    └── README.md
```

Expected training file:
- `create_dataset/data/nsfw_prompts.csv`
- Required columns: `id,original_text,rewritten_text,category`

## 1) Environment Setup

```bash
cd /media02/ltnghia24
cd create_dataset/lora_prompt_enhance
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Hardware Requirements

Recommended:
- GPU NVIDIA with CUDA
- VRAM >= 16GB (for comfortable QLoRA on 7B)
- RAM >= 32GB
- Disk >= 30GB free

Can run with lower resources:
- 12GB VRAM may work by reducing:
  - `--train_batch_size 1`
  - `--grad_accum 16`
  - `--max_seq_len 512`

Not recommended:
- CPU-only training (too slow for practical use)

## 3) Prepare Train/Val Split

This script:
- Reads CSV
- Cleans missing/empty values
- Drops exact duplicate pairs
- Splits into train/val
- Saves HuggingFace dataset on disk

Run inside `create_dataset/lora_prompt_enhance`:

```bash
python prepare_dataset.py \
  --input_csv ../data/nsfw_prompts.csv \
  --output_dir ./artifacts/dataset \
  --val_ratio 0.1 \
  --seed 42
```

Output:
- `./artifacts/dataset` with 2 splits:
  - `train`
  - `test` (used as validation)

## 4) Train QLoRA

Default command:

```bash
python train_lora.py \
  --dataset_dir ./artifacts/dataset \
  --model_name /media02/ltnghia24/models/Qwen2.5-7B-Instruct \
  --output_dir ./artifacts/qwen25-lora \
  --max_seq_len 768 \
  --epochs 4 \
  --learning_rate 1e-4 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --grad_accum 8
```

Training output:
- Checkpoints in `./artifacts/qwen25-lora`
- Final adapter in `./artifacts/qwen25-lora/final_adapter`

### Training prompt format used internally

Each sample is converted to:
- `system`: controlled enhancement instruction
- `user`:
  - `category: <category from CSV>`
  - `raw_prompt: <original_text>`
- `assistant`: `<rewritten_text>`

## 5) Inference

### Input CSV

Script đọc 2 cột từ input CSV:
- `prompt` (hoặc chỉnh qua `--input_col`)
- `category` (hoặc chỉnh qua `--category_col`)

Nếu không có cột `category`, dùng `--default_category`.

### Chạy inference trên i2p_df_final.csv

```bash
CUDA_VISIBLE_DEVICES=7 python inference_enhance.py \
  --adapter_dir ./artifacts/qwen25-lora/final_adapter \
  --input_csv /media02/ltnghia24/create_dataset/data/i2p_df_final.csv \
  --input_col prompt \
  --category_col category \
  --output_csv ./artifacts/enhanced_i2p.csv \
  --max_new_tokens 180 \
  --temperature 0.35 \
  --top_p 0.9
```

### Output CSV

File `enhanced_i2p.csv` có 4 cột:

| Cột | Nội dung |
|---|---|
| `id` | Row index từ input CSV |
| `prompt` | Raw prompt gốc |
| `category` | Category của prompt |
| `enhanced_prompt` | Prompt sau khi enhance |

### Resume sau crash

Script tự động **resume** — nếu chạy lại cùng lệnh, nó đọc `output_csv` đã có, bỏ qua các ID đã xử lý và tiếp tục từ chỗ dừng.

```bash
# Chạy lại y chang lệnh cũ là được, không cần làm gì thêm
CUDA_VISIBLE_DEVICES=7 python inference_enhance.py \
  --adapter_dir ./artifacts/qwen25-lora/final_adapter \
  --input_csv /media02/ltnghia24/create_dataset/data/i2p_df_final.csv \
  --input_col prompt \
  --category_col category \
  --output_csv ./artifacts/enhanced_i2p.csv
```

## 6) Run on SLURM (khuyến nghị)

Dùng khi chạy trên cluster thay vì chạy tay trực tiếp.

### Điều kiện tiên quyết

- Model đã tải về local:
  ```bash
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
      --local-dir /media02/ltnghia24/models/Qwen2.5-7B-Instruct
  ```
- Đã chạy `prepare_dataset.py` và có `./artifacts/dataset/train` trên disk.
- venv tại `/media02/ltnghia24/.venv` đã cài đủ `requirements.txt`.

### Submit job

```bash
cd /media02/ltnghia24/create_dataset/lora_prompt_enhance

# Submit và lấy job ID
sbatch run_training_job.sh
```

### Theo dõi tiến độ

```bash
# Xem trạng thái job
squeue -u $USER

# Theo dõi stdout SLURM realtime (thay <JOB_ID>)
tail -f /media02/ltnghia24/logs/lora_<JOB_ID>.out

# Theo dõi log chi tiết của trainer (ghi bởi train_lora.py)
tail -f /media02/ltnghia24/logs/train_<JOB_ID>.log

# Xem stderr nếu có lỗi
tail -f /media02/ltnghia24/logs/lora_<JOB_ID>.err
```

### Ý nghĩa 2 file log

| File | Nội dung |
|---|---|
| `lora_<JOB_ID>.out` | SLURM stdout: header, GPU info, summary kết quả job |
| `train_<JOB_ID>.log` | Python logger: tiến độ step/epoch, loss, GPU memory, checkpoint |

### Khi job thành công

Adapter được lưu tại:

```text
./artifacts/qwen25-lora/final_adapter/
```

### Khi job bị crash / timeout

- Trainer đã lưu checkpoint mỗi epoch (giữ 2 checkpoint gần nhất).
- Checkpoint nằm tại `./artifacts/qwen25-lora/checkpoint-<step>/`.
- Để resume từ checkpoint:

```bash
# Sửa biến OUTPUT_DIR trong run_training_job.sh trỏ đúng thư mục cũ,
# rồi Transformers sẽ tự detect và resume checkpoint mới nhất.
sbatch run_training_job.sh
```

### Tùy chỉnh tài nguyên

Sửa đầu file `run_training_job.sh`:

```bash
#SBATCH --time=08:00:00     # tăng time limit nếu cần
#SBATCH --mem=64G           # tăng RAM nếu OOM
#SBATCH --gres=gpu:1        # giữ nguyên (1 GPU là đủ)
```

---

## 7) Useful Tips

- If outputs are too repetitive:
  - Increase `temperature` slightly (0.45 to 0.6)
- If outputs drift from original intent:
  - Lower `temperature` (0.2 to 0.35)
  - Reduce epochs or learning rate
- If OOM:
  - Lower `--max_seq_len`
  - Lower train batch size
  - Increase gradient accumulation

## 7) Chỉ định CUDA device

### Chạy tay (không dùng SLURM)

Dùng biến môi trường `CUDA_VISIBLE_DEVICES` trước lệnh python:

```bash
# Dùng GPU 0
CUDA_VISIBLE_DEVICES=0 python train_lora.py \
  --dataset_dir ./artifacts/dataset \
  --model_name /media02/ltnghia24/models/Qwen2.5-7B-Instruct \
  --output_dir ./artifacts/qwen25-lora

# Dùng GPU 1
CUDA_VISIBLE_DEVICES=1 python train_lora.py ...

# Dùng GPU 2 và 3 (nếu muốn multi-GPU)
CUDA_VISIBLE_DEVICES=2,3 python train_lora.py ...
```

### Khi dùng SLURM

SLURM tự gán GPU qua `--gres=gpu:1`, không cần set `CUDA_VISIBLE_DEVICES` thủ công.
Nếu muốn ép GPU cụ thể, thêm vào `run_training_job.sh` sau dòng `set -euo pipefail`:

```bash
export CUDA_VISIBLE_DEVICES=0
```

### Kiểm tra GPU nào đang được dùng

```bash
# Xem GPU nào đang free trước khi chạy
nvidia-smi

# Trong lúc train đang chạy, xem GPU usage realtime
watch -n 2 nvidia-smi
```

---

## 8) Notes

- This project trains adapter weights (LoRA), not full model weights.
- Keep train/inference instruction style consistent for best quality.
