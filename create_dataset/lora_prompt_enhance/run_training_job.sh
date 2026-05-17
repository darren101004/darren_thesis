#!/bin/bash
#SBATCH --job-name=lora_enhance
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/media02/ltnghia24/logs/lora_%j.out
#SBATCH --error=/media02/ltnghia24/logs/lora_%j.err

# ===========================================================================
# Config — chỉnh các biến này nếu cần
# ===========================================================================
VENV="/media02/ltnghia24/.venv"
BASE_DIR="/media02/ltnghia24/create_dataset/lora_prompt_enhance"
LOG_DIR="/media02/ltnghia24/logs"

# Dataset (đã chạy prepare_dataset.py trước)
DATASET_DIR="${BASE_DIR}/artifacts/dataset"

# Model base (local path, downloaded via huggingface-cli)
MODEL_NAME="/media02/ltnghia24/models/Qwen2.5-7B-Instruct"

# Output adapter
OUTPUT_DIR="${BASE_DIR}/artifacts/qwen25-lora"

# Training hyperparameters
MAX_SEQ_LEN=768
EPOCHS=4
LR=1e-4
TRAIN_BATCH=2
EVAL_BATCH=2
GRAD_ACCUM=8
SEED=42

# Log file (ghi riêng ngoài SLURM stdout)
TRAIN_LOG="${LOG_DIR}/train_${SLURM_JOB_ID}.log"

# ===========================================================================
# Setup
# ===========================================================================
set -euo pipefail

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo " Job ID       : ${SLURM_JOB_ID}"
echo " Job name     : ${SLURM_JOB_NAME}"
echo " Node         : $(hostname)"
echo " Started      : $(date)"
echo " Model        : ${MODEL_NAME}"
echo " Dataset dir  : ${DATASET_DIR}"
echo " Output dir   : ${OUTPUT_DIR}"
echo " Train log    : ${TRAIN_LOG}"
echo "============================================================"

# GPU info
echo "[INFO] GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
    --format=csv,noheader

# ===========================================================================
# Activate venv
# ===========================================================================
echo "[INFO] Activating venv: ${VENV}"
source "${VENV}/bin/activate"
echo "[INFO] Python : $(which python) — $(python --version)"
echo "[INFO] Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"

# ===========================================================================
# Sanity checks
# ===========================================================================
if [ ! -d "${MODEL_NAME}" ]; then
    echo "[ERROR] Model not found at ${MODEL_NAME}"
    echo "        Run: huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ${MODEL_NAME}"
    exit 1
fi

echo "[INFO] Model OK: ${MODEL_NAME}"

if [ ! -d "${DATASET_DIR}" ]; then
    echo "[ERROR] Dataset not found at ${DATASET_DIR}"
    echo "        Run prepare_dataset.py first."
    exit 1
fi

if [ ! -d "${DATASET_DIR}/train" ]; then
    echo "[ERROR] Dataset split 'train' missing in ${DATASET_DIR}"
    exit 1
fi

echo "[INFO] Dataset OK: $(ls ${DATASET_DIR})"

# ===========================================================================
# Run training
# ===========================================================================
echo "[INFO] Starting training — $(date)"

python "${BASE_DIR}/train_lora.py" \
    --dataset_dir    "${DATASET_DIR}" \
    --model_name     "${MODEL_NAME}" \
    --output_dir     "${OUTPUT_DIR}" \
    --max_seq_len    "${MAX_SEQ_LEN}" \
    --epochs         "${EPOCHS}" \
    --learning_rate  "${LR}" \
    --train_batch_size "${TRAIN_BATCH}" \
    --eval_batch_size  "${EVAL_BATCH}" \
    --grad_accum     "${GRAD_ACCUM}" \
    --seed           "${SEED}" \
    --log_file       "${TRAIN_LOG}"

EXIT_CODE=$?

# ===========================================================================
# Summary
# ===========================================================================
echo "============================================================"
echo " Job finished : $(date)"
echo " Exit code    : ${EXIT_CODE}"

if [ "${EXIT_CODE}" -eq 0 ]; then
    echo " Result       : SUCCESS"
    ADAPTER_PATH="${OUTPUT_DIR}/final_adapter"
    if [ -d "${ADAPTER_PATH}" ]; then
        echo " Adapter saved: ${ADAPTER_PATH}"
        echo " Adapter files: $(ls ${ADAPTER_PATH})"
    fi
else
    echo " Result       : FAILED"
    echo " Check logs   : ${TRAIN_LOG}"
    echo "               /media02/ltnghia24/logs/lora_${SLURM_JOB_ID}.err"
fi

echo " Full train log: ${TRAIN_LOG}"
echo "============================================================"

exit "${EXIT_CODE}"
