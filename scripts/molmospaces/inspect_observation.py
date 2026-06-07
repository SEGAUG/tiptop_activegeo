#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _default_work() -> Path:
    return Path(os.environ.get("WORK", "/data/data2/jinhui.lin/code/aicode"))


def main() -> None:
    work = _default_work()
    molmo = Path(os.environ.get("MOLMO", work / "molmospaces"))
    tiptop = Path(os.environ.get("TIPTOP", work / "tiptop"))
    if not tiptop.exists():
        tiptop = work / "tiptop_activegeo"
    assets = Path(os.environ.get("MLSPACES_ASSETS_DIR", work / "molmospaces_assets"))

    parser = argparse.ArgumentParser(description="Run one MolmoSpaces step and print observation schema.")
    parser.add_argument(
        "--benchmark_dir",
        default=str(
            assets
            / "benchmarks/molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
        ),
    )
    parser.add_argument("--output_dir", default=str(work / "molmospaces_runs/inspect_observation"))
    parser.add_argument("--schema_path", default=str(tiptop / "logs/observation_schema.json"))
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("MLSPACES_ASSETS_DIR", str(assets))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["TIPTOP_MOLMOSPACES_LOG_DIR"] = str(Path(args.schema_path).parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tiptop), str(molmo), env.get("PYTHONPATH", "")]
    )

    cmd = [
        sys.executable,
        str(molmo / "molmo_spaces/evaluation/eval_main.py"),
        "tiptop_molmospaces.configs:TiPToPEvalConfig",
        "--benchmark_dir",
        args.benchmark_dir,
        "--output_dir",
        args.output_dir,
        "--task_horizon_steps",
        "1",
        "--idx",
        str(args.idx),
        "--no_wandb",
    ]
    subprocess.run(cmd, cwd=str(molmo), env=env, check=True)

    schema_path = Path(args.schema_path)
    schema = json.loads(schema_path.read_text())
    summary = schema.get("summary", {})
    print(f"observation type: {summary.get('observation_type')}")
    print(f"top-level keys: {summary.get('top_level_keys')}")
    print(f"camera image keys: {summary.get('camera_image_keys')}")
    for name, item in summary.get("images", {}).items():
        print(f"image {name}: shape={item.get('shape')} dtype={item.get('dtype')}")
    print(f"has depth: {summary.get('has_depth')}")
    print(f"robot_state keys: {summary.get('robot_state_keys')}")
    print(f"qpos shape: {summary.get('qpos_shape')}")
    print(f"task/language: {summary.get('task_or_language')}")
    print(f"schema saved: {schema_path}")


if __name__ == "__main__":
    main()
