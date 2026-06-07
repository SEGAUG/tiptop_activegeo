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
export TIPTOP_ENABLE_LIVE_PLANNING=1
export TIPTOP_REQUIRE_PLAN=1
export TIPTOP_VLM_BACKEND="${TIPTOP_VLM_BACKEND:-qwen}"
export TIPTOP_ENABLE_RENDERED_DEPTH=1
export TIPTOP_ENABLE_QWEN_GROUNDING=1
export TIPTOP_ENABLE_SAM_MASKS=1
export TIPTOP_MOLMOSPACES_LOG_DIR="${TIPTOP_MOLMOSPACES_LOG_DIR:-$TIPTOP/logs}"
export PYTHONPATH="$TIPTOP:$MOLMO:${PYTHONPATH:-}"

BENCHMARK_DIR="${BENCHMARK_DIR:-$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/molmospaces_runs/smoke_pick_v15_live_qwen_sam}"

cd "$MOLMO"
python molmo_spaces/evaluation/eval_main.py \
  tiptop_molmospaces.configs:TiPToPDepthEvalConfig \
  --benchmark_dir "$BENCHMARK_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --task_horizon_steps 500 \
  --idx 0 \
  --no_wandb
