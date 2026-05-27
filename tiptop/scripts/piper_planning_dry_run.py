from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

from tiptop.motion_planning import build_curobo_solvers
from tiptop.piper import PiperRobotClient
from tiptop.piper.native_executor import build_joint_trajectory_report


def piper_planning_dry_run(
    bridge_url: str = "http://10.31.3.54:8766",
    output_dir: str = "/data/data2/jinhui.lin/code/aicode/piper_real_outputs/planning_dry_run",
    joint_index: int = 5,
    delta_rad: float = 0.01,
    waypoint_dt_s: float = 5.0,
) -> None:
    """Plan a tiny Piper joint-space motion with cuRobo and save it without moving the robot."""
    if abs(delta_rad) > np.deg2rad(1.0):
        raise ValueError("delta_rad must be <= 1 degree for planning dry-run")

    client = PiperRobotClient(base_url=bridge_url, min_waypoint_dt_s=waypoint_dt_s)
    health = client.health()
    if health.get("motion_allowed") is True:
        raise RuntimeError("Refusing planning dry-run while bridge motion gate is open")

    q_current = client.get_joint_positions().astype(float)
    q_target = q_current.copy()
    q_target[joint_index] += float(delta_rad)

    _, motion_gen, _ = build_curobo_solvers(
        num_particles=32,
        num_spheres=16,
        collision_activation_distance=0.01,
        include_workspace=True,
    )
    tensor_args = TensorDeviceType()
    js_start = JointState.from_position(tensor_args.to_device(q_current))
    js_target = JointState.from_position(tensor_args.to_device(q_target))
    result = motion_gen.plan_single_js(js_start[None], js_target[None], MotionGenPlanConfig(time_dilation_factor=0.1))
    if not result.success.all():
        raise RuntimeError(f"cuRobo failed to plan tiny Piper dry-run motion: {result.status}")

    positions = result.interpolated_plan.position.detach().cpu().numpy()
    durations = np.full((positions.shape[0],), waypoint_dt_s, dtype=float)
    safety = build_joint_trajectory_report(q_current, positions, durations, min_waypoint_dt_s=waypoint_dt_s)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(output_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / "trajectory.npz", positions=positions, durations=durations)
    payload = {
        "timestamp": timestamp,
        "bridge_url": bridge_url,
        "health": health,
        "q_current": q_current.tolist(),
        "q_target": q_target.tolist(),
        "joint_index": int(joint_index),
        "delta_rad": float(delta_rad),
        "trajectory_npz": str(save_dir / "trajectory.npz"),
        "safety": safety,
        "executed": False,
    }
    (save_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"save_dir": str(save_dir), **payload}, indent=2))
    torch.cuda.empty_cache()


def entrypoint() -> None:
    tyro.cli(piper_planning_dry_run)


if __name__ == "__main__":
    entrypoint()
