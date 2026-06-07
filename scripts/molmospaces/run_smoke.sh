#!/usr/bin/env bash
set -euo pipefail

export WORK="${WORK:-/data/data2/jinhui.lin/code/aicode}"
export MOLMO="${MOLMO:-$WORK/molmospaces}"
export TIPTOP="${TIPTOP:-$WORK/tiptop}"
if [ ! -d "$TIPTOP" ]; then
  export TIPTOP="$WORK/tiptop_activegeo"
fi
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$WORK/molmospaces_assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$TIPTOP:$MOLMO:${PYTHONPATH:-}"

BENCHMARK_DIR="${BENCHMARK_DIR:-$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/molmospaces_runs/smoke_pick_v15}"

cd "$MOLMO"
python molmo_spaces/evaluation/eval_main.py \
  tiptop_molmospaces.configs:TiPToPEvalConfig \
  --benchmark_dir "$BENCHMARK_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --task_horizon_steps 500 \
  --idx 0 \
  --no_wandb
