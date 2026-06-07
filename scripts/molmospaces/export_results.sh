#!/usr/bin/env bash
set -euo pipefail

export WORK="${WORK:-/data/data2/jinhui.lin/code/aicode}"
export MOLMO="${MOLMO:-$WORK/molmospaces}"
RUN_ROOT="${RUN_ROOT:-$WORK/molmospaces_runs}"
CSV_DIR="${CSV_DIR:-$WORK/molmospaces_submission_csv}"
POLICY_NAME="${POLICY_NAME:-TiPToP-ActiveGeometry}"
SUCCESS_CONDITION="${SUCCESS_CONDITION:-oracle}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ZIP_PATH="$WORK/TiPToP-ActiveGeometry_molmospaces_results_${TIMESTAMP}.zip"

mkdir -p "$CSV_DIR"

shopt -s nullglob
for run_dir in "$RUN_ROOT"/*; do
  [ -d "$run_dir" ] || continue
  if ! find "$run_dir" -name '*.h5' -print -quit | grep -q .; then
    echo "Skipping $run_dir: no .h5 files"
    continue
  fi
  task_name="$(basename "$run_dir")"
  csv_path="$CSV_DIR/${task_name}.csv"
  python "$MOLMO/scripts/benchmarks/eval_to_csv.py" \
    "$run_dir" \
    "$POLICY_NAME" \
    --success-condition "$SUCCESS_CONDITION" \
    --output-csv "$csv_path"
done

(cd "$CSV_DIR" && zip -r "$ZIP_PATH" .)
echo "$ZIP_PATH"
