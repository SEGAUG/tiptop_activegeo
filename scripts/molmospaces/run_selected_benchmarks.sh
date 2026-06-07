#!/usr/bin/env bash
set -euo pipefail

export WORK="${WORK:-/data/data2/jinhui.lin/code/aicode}"
export MOLMO="${MOLMO:-$WORK/molmospaces}"
export TIPTOP="${TIPTOP:-$WORK/tiptop}"
if [ ! -d "$TIPTOP" ]; then
  export TIPTOP="$WORK/tiptop_activegeo"
fi
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$WORK/molmospaces_assets}"
export OUT="${OUT:-$WORK/molmospaces_runs}"
export POLICY_CONFIG="${POLICY_CONFIG:-tiptop_molmospaces.configs:TiPToPEvalConfig}"
export TASK_HORIZON="${TASK_HORIZON:-500}"
export IDX_ARG="${IDX_ARG:---idx 0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$TIPTOP:$MOLMO:${PYTHONPATH:-}"

BENCH_ROOT="$MLSPACES_ASSETS_DIR/benchmarks"

declare -A BENCHMARKS=(
  [open_v1]="$BENCH_ROOT/molmospaces-bench-v1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark"
  [close_v1]="$BENCH_ROOT/molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark"
  [pick_v1]="$BENCH_ROOT/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
  [pnp_v1]="$BENCH_ROOT/molmospaces-bench-v1/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark"
  [pick_v15]="$BENCH_ROOT/molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
  [pick_v2_classic]="$BENCH_ROOT/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark"
  [pick_v2_filament]="$BENCH_ROOT/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark"
  [pick_v2_randcam]="$BENCH_ROOT/molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark"
  [pnp_v2]="$BENCH_ROOT/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark"
  [pnp_next_to_v2]="$BENCH_ROOT/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark"
  [pnp_color_v2]="$BENCH_ROOT/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark"
)

declare -A EXTRA_ARGS=(
  [pick_v2_filament]="--use-filament"
  [pick_v2_randcam]="--use_eval_cameras --camera_rand_level 100"
)

TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(open_v1 close_v1 pick_v1 pnp_v1 pick_v15 pick_v2_classic pick_v2_filament pick_v2_randcam pnp_v2 pnp_next_to_v2 pnp_color_v2)
fi

cd "$MOLMO"
for task in "${TASKS[@]}"; do
  benchmark_dir="${BENCHMARKS[$task]:-}"
  if [ -z "$benchmark_dir" ]; then
    echo "Unknown benchmark task: $task" >&2
    exit 2
  fi
  if [ ! -d "$benchmark_dir" ]; then
    echo "Benchmark directory not found: $benchmark_dir" >&2
    exit 3
  fi
  echo "Running $task"
  # shellcheck disable=SC2086
  python molmo_spaces/evaluation/eval_main.py \
    "$POLICY_CONFIG" \
    --benchmark_dir "$benchmark_dir" \
    --output_dir "$OUT/$task" \
    --task_horizon_steps "$TASK_HORIZON" \
    --no_wandb \
    ${IDX_ARG} \
    ${EXTRA_ARGS[$task]:-}
done
