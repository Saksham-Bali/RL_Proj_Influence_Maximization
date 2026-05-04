#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR/touplegdd"
ARTIFACT_DIR="$ROOT_DIR/artifacts"
LOG_DIR="$ARTIFACT_DIR/logs"
OUT_DIR="$ARTIFACT_DIR/output"
VENV_DIR="$ROOT_DIR/.venv"

mkdir -p "$LOG_DIR" "$OUT_DIR"

RUN_TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/run_${RUN_TS}.log"

INFERENCE_ONLY=0
MODEL_FILE_OVERRIDE=""

print_usage() {
  cat <<'USAGE'
Usage: bash run.sh [--inference-only] [--model-file <relative_model_path>]

Options:
  --inference-only         Skip training and run inference only.
  --model-file <path>      Relative path (from touplegdd/) to checkpoint for inference.
  -h, --help               Show this help message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inference-only)
      INFERENCE_ONLY=1
      shift
      ;;
    --model-file)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --model-file requires a path argument."
        exit 1
      fi
      MODEL_FILE_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Mirror all stdout/stderr to a timestamped logfile.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Starting end-to-end pipeline at $(date -Iseconds)"
echo "[INFO] Repository root: $ROOT_DIR"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] Missing project directory: $PROJECT_DIR"
  exit 1
fi

# ---------- System bootstrap (Ubuntu-friendly) ----------
if ! command -v python3 >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    echo "[INFO] python3 not found. Installing python3, pip, and venv via apt-get..."
    apt-get update
    apt-get install -y python3 python3-pip python3-venv
  else
    echo "[ERROR] python3 is not installed and apt-get is unavailable."
    exit 1
  fi
fi

# Ensure venv module exists (important for minimal Ubuntu images)
if ! python3 -m venv --help >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    echo "[INFO] python3-venv missing. Installing via apt-get..."
    apt-get update
    apt-get install -y python3-venv
  else
    echo "[ERROR] python3-venv missing and apt-get is unavailable."
    exit 1
  fi
fi

# ---------- Virtual environment ----------
echo "[INFO] Creating virtual environment at $VENV_DIR"
python3 -m venv --clear "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ---------- Python dependencies ----------
echo "[INFO] Installing core dependencies"
pip install --no-cache-dir numpy scipy matplotlib tqdm

TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
CUDA_TAG="${CUDA_TAG:-auto}"
DETECTED_CUDA_VERSION=""
GPU_HINT=0

if [[ "$CUDA_TAG" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_HINT=1
    DETECTED_CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1 || true)"
    if [[ -n "$DETECTED_CUDA_VERSION" ]]; then
      cuda_major="${DETECTED_CUDA_VERSION%%.*}"
      cuda_minor="${DETECTED_CUDA_VERSION#*.}"

      if (( cuda_major > 12 || (cuda_major == 12 && cuda_minor >= 4) )); then
        CUDA_TAG="cu124"
      elif (( cuda_major == 12 && cuda_minor >= 1 )); then
        CUDA_TAG="cu121"
      elif (( cuda_major == 11 && cuda_minor >= 8 )) || (( cuda_major > 11 )); then
        CUDA_TAG="cu118"
      else
        CUDA_TAG="cpu"
      fi
    else
      CUDA_TAG="cu118"
    fi
  elif [[ -e /dev/nvidia0 || -e /dev/nvidiactl || -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then
    # GPU appears to be passed through, but nvidia-smi may be unavailable.
    # Use the most compatible CUDA wheel by default.
    GPU_HINT=1
    CUDA_TAG="cu118"
  else
    CUDA_TAG="cpu"
  fi
fi

if [[ "$CUDA_TAG" == "cpu" ]]; then
  echo "[INFO] Installing CPU PyTorch (${TORCH_VERSION})"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
  echo "[INFO] Installing CUDA-enabled PyTorch (${TORCH_VERSION}, ${CUDA_TAG})"
  if [[ -n "$DETECTED_CUDA_VERSION" ]]; then
    echo "[INFO] nvidia-smi detected CUDA capability: ${DETECTED_CUDA_VERSION}"
  fi
  TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
fi

pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"

PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
echo "[INFO] Installing torch_scatter from: ${PYG_WHL_URL}"
pip install --no-cache-dir torch_scatter -f "$PYG_WHL_URL"

echo "[INFO] Installing torch_geometric"
pip install --no-cache-dir torch_geometric

echo "[INFO] Torch/CUDA runtime check"
CUDA_RUNTIME_OK=0
python - <<'PY'
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Torch CUDA runtime: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

if python - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  CUDA_RUNTIME_OK=1
fi

if [[ "$CUDA_TAG" != "cpu" && "$GPU_HINT" -eq 1 && "$CUDA_RUNTIME_OK" -eq 0 ]]; then
  echo "[WARN] CUDA wheel installed but GPU runtime is not usable in this container."
  echo "[WARN] If you expected GPU, run the container with NVIDIA runtime (e.g. --gpus all)."
fi

# ---------- Pipeline execution ----------
cd "$PROJECT_DIR"

# Configurable knobs (can be overridden by environment variables)
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20000}"
TRAIN_BUDGET="${TRAIN_BUDGET:-5}"
TRAIN_ALPHA="${TRAIN_ALPHA:-1.0}"
TRAIN_BETA="${TRAIN_BETA:-2.0}"
TRAIN_BS="${TRAIN_BS:-16}"
TRAIN_NSTEP="${TRAIN_NSTEP:-2}"

REQUIRED_FILES=(test_graph_new.txt test_comms_new.txt main.py inference.py)
if [[ "$INFERENCE_ONLY" -eq 0 ]]; then
  REQUIRED_FILES+=(train_graphs_new train_comms_new)
fi

for required in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Required file/path not found: $PROJECT_DIR/$required"
    exit 1
  fi
done

if [[ "$INFERENCE_ONLY" -eq 0 ]]; then
  echo "[INFO] Running training"
  python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"[INFO] CUDA detected. Training will run on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] CUDA not detected. Training will run on CPU.")
PY

  python main.py \
    --graph train_graphs_new \
    --community_path train_comms_new \
    --budget "$TRAIN_BUDGET" \
    --alpha "$TRAIN_ALPHA" \
    --beta "$TRAIN_BETA" \
    --bs "$TRAIN_BS" \
    --epoch "$TRAIN_EPOCHS" \
    --model Tripling \
    --model_file tripling.ckpt \
    --n_step "$TRAIN_NSTEP"

  LATEST_MODEL="$(python - <<'PY'
import glob
import os
candidates = [p for p in glob.glob('*/tripling.ckpt') if os.path.isfile(p)]
if not candidates:
    raise SystemExit('No trained model file */tripling.ckpt found after training.')
print(max(candidates, key=os.path.getmtime))
PY
)"
else
  echo "[INFO] Inference-only mode enabled. Training step skipped."
  if [[ -n "$MODEL_FILE_OVERRIDE" ]]; then
    if [[ ! -f "$MODEL_FILE_OVERRIDE" ]]; then
      echo "[ERROR] Model file not found: $PROJECT_DIR/$MODEL_FILE_OVERRIDE"
      exit 1
    fi
    LATEST_MODEL="$MODEL_FILE_OVERRIDE"
  else
    LATEST_MODEL="$(python - <<'PY'
import glob
import os
candidates = [p for p in glob.glob('**/tripling.ckpt*', recursive=True)
              if os.path.isfile(p)]
if not candidates:
    raise SystemExit(
        'No checkpoint found. Provide one with --model-file <path>.')
print(max(candidates, key=os.path.getmtime))
PY
)"
  fi
fi

echo "[INFO] Selected model checkpoint: $LATEST_MODEL"

echo "[INFO] Running inference"
python inference.py \
  --graph test_graph_new.txt \
  --community_path test_comms_new.txt \
  --model Tripling \
  --model_file "$LATEST_MODEL" \
  --num_communities 1 \
  --budget "$TRAIN_BUDGET"

# Baseline evaluation removed


# ---------- Collect artifacts ----------
MODEL_RUN_DIR="$(dirname "$LATEST_MODEL")"
TARGET_RUN_DIR="$OUT_DIR/$RUN_TS"
mkdir -p "$TARGET_RUN_DIR"

if [[ "$MODEL_RUN_DIR" == "." ]]; then
  mkdir -p "$TARGET_RUN_DIR/model_run"
  cp "$LATEST_MODEL" "$TARGET_RUN_DIR/model_run/"
else
  cp -r "$MODEL_RUN_DIR" "$TARGET_RUN_DIR/model_run"
fi

# Copy common artifacts if present.
for f in training_diagnostics.png average_metrics.png list_cumul_reward.txt; do
  if [[ -f "$MODEL_RUN_DIR/$f" ]]; then
    cp "$MODEL_RUN_DIR/$f" "$TARGET_RUN_DIR/"
  fi
done

cp "$LOG_FILE" "$TARGET_RUN_DIR/pipeline.log"

cat > "$TARGET_RUN_DIR/README_ARTIFACTS.txt" <<TXT
Run timestamp: $RUN_TS
Model directory: $MODEL_RUN_DIR
Primary log: $LOG_FILE

This folder contains:
- model checkpoints and outputs from training
- plots generated by runner.py (if produced)
- full pipeline log (pipeline.log)
TXT

echo "[INFO] Pipeline completed successfully at $(date -Iseconds)"
echo "[INFO] Artifacts saved in: $TARGET_RUN_DIR"
