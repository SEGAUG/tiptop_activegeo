from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro

from tiptop.piper import PiperRobotClient
from tiptop.piper.native_executor import build_joint_trajectory_report


def piper_native_execution_dry_run(
    bridge_url: str = "http://10.31.3.54:8766",
    trajectory_npz: str = "",
    output_path: str = "piper_real_outputs/native_execution_dry_run_report.json",
    min_waypoint_dt_s: float = 3.0,
) -> None:
    """Validate a TiPToP/curobo Piper joint trajectory without moving the robot."""
    if not trajectory_npz:
        raise ValueError("trajectory_npz is required and must contain positions and durations arrays")

    client = PiperRobotClient(base_url=bridge_url, min_waypoint_dt_s=min_waypoint_dt_s)
    health = client.health()
    if health.get("motion_allowed") is True:
        raise RuntimeError("Refusing dry-run while remote bridge motion gate is open")

    q_current = client.get_joint_positions()
    with np.load(trajectory_npz) as data:
        positions = data["positions"]
        durations = data["durations"]

    report = build_joint_trajectory_report(q_current, positions, durations, min_waypoint_dt_s=min_waypoint_dt_s)
    payload = {
        "bridge_url": bridge_url,
        "trajectory_npz": trajectory_npz,
        "health": health,
        "q_current": q_current.tolist(),
        "report": report,
        "would_execute": False,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(path), **payload}, indent=2))


def entrypoint() -> None:
    tyro.cli(piper_native_execution_dry_run)


if __name__ == "__main__":
    entrypoint()
