#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TiPToP MolmoSpaces depth observation artifacts.")
    parser.add_argument(
        "--log-dir",
        default="logs/molmospaces_live",
        help="Live MolmoSpaces log directory produced by the TiPToP adapter.",
    )
    parser.add_argument("--episode", default="episode_000000", help="Episode directory name.")
    args = parser.parse_args()

    episode_dir = Path(args.log_dir) / args.episode
    observation_summary = _load_json(episode_dir / "observation_summary.json")
    depth_summary = _load_json(episode_dir / "depth_summary.json")
    planning_failure = _load_json(episode_dir / "planning_failure.json")

    payload = {
        "episode_dir": str(episode_dir),
        "camera_image_keys": observation_summary.get("camera_image_keys", []),
        "depth_keys": depth_summary.get("depth_keys", observation_summary.get("depth_keys", [])),
        "intrinsics_keys": depth_summary.get("intrinsics_keys", observation_summary.get("camera_intrinsics_keys", [])),
        "depth_available": depth_summary.get("depth_available", False),
        "xyz_map_created": depth_summary.get("xyz_map_created", False),
        "xyz_keys": depth_summary.get("xyz_keys", []),
        "depth_backend": depth_summary.get("depth_backend", observation_summary.get("depth_backend")),
        "failure_reason": planning_failure.get("failure_reason"),
        "failure_layer": planning_failure.get("debug", {}).get("failure_layer"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
