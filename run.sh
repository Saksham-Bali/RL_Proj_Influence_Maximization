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

install_torch_stack() {
  local target_tag="$1"
  local torch_index_url
  local pyg_url
  local torch_base
  local pyg_tag

  if [[ "$target_tag" == "cpu" ]]; then
    torch_index_url="https://download.pytorch.org/whl/cpu"
  else
    torch_index_url="https://download.pytorch.org/whl/${target_tag}"
  fi

  echo "[INFO] Trying torch==${TORCH_VERSION} (${target_tag})"
  if ! pip install --no-cache-dir --force-reinstall \
    "torch==${TORCH_VERSION}" \
    --index-url "$torch_index_url"; then
    echo "[WARN] Torch install failed for ${target_tag}."
    return 1
  fi

  if ! python - <<'PY' >/dev/null 2>&1
import torch
_ = torch.__version__
PY
  then
    echo "[WARN] Torch import failed after install (${target_tag})."
    return 1
  fi

  torch_base="$(python - <<'PY'
import torch
base = torch.__version__.split('+')[0]
cuda = torch.version.cuda
if cuda:
    major, minor = cuda.split('.')[:2]
    print(base)
    print(f"cu{major}{minor}")
else:
    print(base)
    print("cpu")
PY
)"
  torch_base="$(echo "$torch_base" | sed -n '1p')"
  pyg_tag="$(python - <<'PY'
import torch
cuda = torch.version.cuda
if cuda:
    major, minor = cuda.split('.')[:2]
    print(f"cu{major}{minor}")
else:
    print("cpu")
PY
)"
  pyg_url="https://data.pyg.org/whl/torch-${torch_base}+${pyg_tag}.html"

  echo "[INFO] Installing torch_scatter from: ${pyg_url}"
  if ! pip install --no-cache-dir --force-reinstall torch_scatter -f "$pyg_url"; then
    echo "[WARN] torch_scatter install failed for ${target_tag}."
    return 1
  fi

  return 0
}

GPU_VISIBLE=0
if command -v nvidia-smi >/dev/null 2>&1 || [[ -e /dev/nvidia0 || -e /dev/nvidiactl || -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then
  GPU_VISIBLE=1
fi

INSTALLED=0
if [[ "$CUDA_TAG" != "auto" ]]; then
  if install_torch_stack "$CUDA_TAG"; then
    INSTALLED=1
  else
    echo "[WARN] Requested CUDA_TAG=${CUDA_TAG} failed. Falling back to CPU."
    install_torch_stack cpu
    INSTALLED=1
  fi
else
  if [[ "$GPU_VISIBLE" -eq 1 ]]; then
    for tag in cu124 cu121 cu118; do
      if install_torch_stack "$tag"; then
        INSTALLED=1
        break
      fi
    done
  fi
  if [[ "$INSTALLED" -eq 0 ]]; then
    echo "[INFO] Using CPU fallback torch build."
    install_torch_stack cpu
    INSTALLED=1
  fi
fi

echo "[INFO] Installing torch_geometric"
pip install --no-cache-dir --force-reinstall torch_geometric

echo "[INFO] Torch/CUDA runtime check"
python - <<'PY'
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Torch CUDA runtime: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

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
