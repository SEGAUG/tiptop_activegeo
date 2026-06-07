#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tiptop_molmospaces.live_perception import run_live_perception


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay TiPToP MolmoSpaces Qwen/SAM/object-cloud perception from debug artifacts.")
    parser.add_argument("--episode-dir", default="logs/molmospaces_live/episode_000000")
    parser.add_argument("--camera", default=None)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    summary = _load_json(episode_dir / "observation_summary.json")
    camera = args.camera or "exo_camera_1"

    rgb_path = episode_dir / f"rgb_{camera}.png"
    xyz_path = episode_dir / f"xyz_{camera}.npy"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB artifact not found: {rgb_path}")
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ artifact not found: {xyz_path}")

    observation = {
        camera: np.asarray(Image.open(rgb_path).convert("RGB")),
        "_tiptop_xyz_maps": {camera: np.load(xyz_path)},
        "task_text": summary.get("task_text") or summary.get("task_or_language"),
    }
    result = run_live_perception(observation, episode_dir)
    print(
        json.dumps(
            {
                "success": result.get("success"),
                "stage": result.get("stage"),
                "reason": result.get("reason"),
                "task_text": result.get("task_text"),
                "primary_camera": result.get("primary_camera"),
                "num_detections": len(result.get("detections") or []),
                "num_masks": len(result.get("masks") or []),
                "object_cloud_created": (result.get("object_clouds") or {}).get("object_cloud_created", False),
                "target_object_points": (result.get("object_clouds") or {}).get("target_object_points", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
