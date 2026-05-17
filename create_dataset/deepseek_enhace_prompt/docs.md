# DeepSeek Prompt Augmentation — Setup & Usage Guide

## Overview

`enhace_prompt/data/enhance_prompts_with_deepseek.py` augments unsafe prompts to higher intensity
using a **local** DeepSeek model loaded directly via `transformers` (no vLLM server required).

This guide covers:
1. Create environment with `venv`
2. Download model from Hugging Face
3. Run script with correct input CSV format

---

## 1) Create Virtual Environment

```bash
cd /media02/ltnghia24
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install required packages:

```bash
pip install torch transformers pandas
```

If using GPU, install the CUDA-compatible PyTorch wheel for your driver.

---

## 2) Download DeepSeek Model from Hugging Face

Default model local path used by script:
- `/media02/ltnghia24/models/DeepSeek-R1-Distill-Llama-8B`

Download:

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --local-dir /media02/ltnghia24/models/DeepSeek-R1-Distill-Llama-8B
```

If you use another location, pass it via `--model-path`.

---

## 3) Input CSV Format (Required)

Input file **must** contain these required columns:

| Column | Required | Meaning |
|---|---|---|
| `text` | yes | Raw prompt to be intensified |
| `label` | yes | Label to carry through to output |
| `category` | yes | Category used to pick category-specific system prompt |

Example input:

```csv
text,label,category
"a violent street scene",unsafe,physical harm and violence
"hate speech against group X",unsafe,hate content
```

If one of these columns is missing, script exits with error.

---

## 4) Run Script

From workspace root:

```bash
cd /media02/ltnghia24
source .venv/bin/activate

python enhace_prompt/data/enhance_prompts_with_deepseek.py \
  --input enhace_prompt/data/data.csv \
  --output enhace_prompt/data/processed_data.csv \
  --model-path /media02/ltnghia24/models/DeepSeek-R1-Distill-Llama-8B \
  --max-tokens 500 \
  --temperature 1.2 \
  --log-level INFO \
  --log-file logs/deepseek_augment.log
```

Run on a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python enhace_prompt/data/enhance_prompts_with_deepseek.py ...
```

---

## 5) Resume / Crash Safety

The script writes output in append mode and can resume:
- It reads existing output CSV
- Skips rows whose `original_text` already exists
- Continues from remaining rows

So if crash happens, rerun the same command.

---

## 6) Output CSV Format

Output columns:

| Column | Description |
|---|---|
| `text` | Augmented prompt |
| `original_text` | Original input prompt |
| `label` | Copied from input |
| `category` | Copied from input |

---

## 7) Useful Flags

| Flag | Description |
|---|---|
| `--num-samples N` | Process first N rows after resume-skip |
| `--category CAT` | Process only one category |
| `--dry-run` | Print first prepared prompt and exit |
| `--log-file PATH` | Mirror logs to file |

---

## 8) Quick Troubleshooting

- **Model path not found**: verify `--model-path` directory exists and contains downloaded files.
- **CUDA not detected**: check `torch.cuda.is_available()` and install correct PyTorch CUDA wheel.
- **Out of memory**: reduce `--max-tokens` or run on larger GPU.
- **DeepSeek outputs `<think>...</think>`**: script already strips it via `strip_thinking()`.
