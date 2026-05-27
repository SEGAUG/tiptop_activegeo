from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch
import tyro
from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml
from cutamp.robots import load_piper_container, load_piper_rerun
from cutamp.robots.piper import get_piper_kinematics_model

from tiptop.piper import PiperRobotClient
from tiptop.piper.asset_validation import validate_asset_debug_payload


def _link_name_for_sphere_index(kin_model, sphere_index: int) -> str:
    cfg = kin_model.kinematics_config
    link_index = int(cfg.link_sphere_idx_map.detach().cpu()[sphere_index])
    index_to_name = {int(v): k for k, v in cfg.link_name_to_idx_map.items()}
    return index_to_name.get(link_index, str(link_index))


def _load_self_collision_rules() -> tuple[dict[str, set[str]], dict[str, float]]:
    cfg = load_yaml("cutamp/cutamp/robots/assets/piper/piper.yml")["robot_cfg"]["kinematics"]
    ignore = {k: set(v) for k, v in cfg.get("self_collision_ignore", {}).items()}
    buffer = {k: float(v) for k, v in cfg.get("self_collision_buffer", {}).items()}
    return ignore, buffer


def _collision_pair_report(spheres: torch.Tensor, kin_model, margin_threshold_m: float) -> list[dict[str, Any]]:
    ignore, buffer = _load_self_collision_rules()
    rows: list[dict[str, Any]] = []
    spheres_cpu = spheres.detach().cpu()
    for i in range(spheres_cpu.shape[0]):
        for j in range(i + 1, spheres_cpu.shape[0]):
            link_i = _link_name_for_sphere_index(kin_model, i)
            link_j = _link_name_for_sphere_index(kin_model, j)
            if link_i == link_j:
                continue
            if link_j in ignore.get(link_i, set()) or link_i in ignore.get(link_j, set()):
                continue
            distance_m = float(torch.linalg.norm(spheres_cpu[i, :3] - spheres_cpu[j, :3]))
            threshold_m = (
                float(spheres_cpu[i, 3])
                + float(spheres_cpu[j, 3])
                + buffer.get(link_i, 0.0)
                + buffer.get(link_j, 0.0)
            )
            margin_m = distance_m - threshold_m
            if margin_m <= margin_threshold_m:
                rows.append(
                    {
                        "margin_m": margin_m,
                        "link_i": link_i,
                        "link_j": link_j,
                        "sphere_i": i,
                        "sphere_j": j,
                        "distance_m": distance_m,
                        "threshold_m": threshold_m,
                        "radius_i_m": float(spheres_cpu[i, 3]),
                        "radius_j_m": float(spheres_cpu[j, 3]),
                        "center_i": [float(x) for x in spheres_cpu[i, :3]],
                        "center_j": [float(x) for x in spheres_cpu[j, :3]],
                    }
                )
    return sorted(rows, key=lambda x: x["margin_m"])


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Piper Asset Debug Report",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- bridge_url: `{payload['bridge_url']}`",
        f"- q_current: `{[round(x, 6) for x in payload['q_current']]}`",
        f"- ee_position_m: `{[round(x, 6) for x in payload['ee_position_m']]}`",
        f"- tool_position_m: `{[round(x, 6) for x in payload['tool_position_m']]}`",
        f"- link_sphere_count: `{payload['link_sphere_count']}`",
        f"- gripper_sphere_count: `{payload['gripper_sphere_count']}`",
        f"- asset_ready_for_planning_dry_run: `{payload['validation']['asset_ready_for_planning_dry_run']}`",
        f"- validation_reasons: `{payload['validation']['reasons']}`",
        "",
        "## Near Self-Collision Pairs",
        "",
        "| margin_m | links | spheres | distance_m | threshold_m | radii_m |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in payload["near_pairs"]:
        lines.append(
            f"| {row['margin_m']:.5f} | `{row['link_i']}` / `{row['link_j']}` | "
            f"{row['sphere_i']} / {row['sphere_j']} | {row['distance_m']:.5f} | "
            f"{row['threshold_m']:.5f} | {row['radius_i_m']:.4f} / {row['radius_j_m']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def piper_asset_debug(
    bridge_url: str = "http://10.31.3.54:8766",
    output_dir: str = "/data/data2/jinhui.lin/code/aicode/piper_real_outputs/asset_debug",
    margin_threshold_m: float = 0.03,
    rr_spawn: bool = False,
    log_rrd: bool = True,
) -> None:
    """Debug Piper cuRobo asset at the current real-robot joint feedback without moving the robot."""
    client = PiperRobotClient(base_url=bridge_url)
    health = client.health()
    if health.get("motion_allowed"):
        raise RuntimeError("Refusing to debug asset while Piper bridge motion is enabled.")

    q_current = client.get_joint_positions().astype(float)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(output_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    tensor_args = TensorDeviceType()
    q = tensor_args.to_device(q_current)[None]
    kin_model = get_piper_kinematics_model()
    container = load_piper_container(tensor_args)
    state = kin_model.get_state(q)
    spheres = state.link_spheres_tensor[0]
    gripper_spheres = container.gripper_spheres.detach().cpu()
    world_from_ee = state.ee_pose.get_numpy_matrix()[0]
    tool_from_ee = container.tool_from_ee.detach().cpu().numpy()
    world_from_tool = world_from_ee @ tool_from_ee
    near_pairs = _collision_pair_report(spheres, kin_model, margin_threshold_m)

    payload = {
        "timestamp": timestamp,
        "bridge_url": bridge_url,
        "health": health,
        "q_current": q_current.tolist(),
        "ee_position_m": world_from_ee[:3, 3].tolist(),
        "tool_position_m": world_from_tool[:3, 3].tolist(),
        "world_from_ee": world_from_ee.tolist(),
        "world_from_tool": world_from_tool.tolist(),
        "link_sphere_count": int(spheres.shape[0]),
        "gripper_sphere_count": int(gripper_spheres.shape[0]),
        "near_pairs": near_pairs,
    }
    payload["validation"] = validate_asset_debug_payload(payload)
    (save_dir / "report.json").write_text(json.dumps(payload, indent=2))
    _write_markdown_report(save_dir / "report.md", payload)

    if rr_spawn or log_rrd:
        rr.init("piper_asset_debug", spawn=rr_spawn)
        if log_rrd:
            rr.save(str(save_dir / "piper_asset_debug.rrd"))
        robot_rr = load_piper_rerun(load_mesh=True)
        robot_rr.set_joint_positions(q_current)
        spheres_np = spheres.detach().cpu().numpy()
        rr.log("piper/link_spheres", rr.Points3D(positions=spheres_np[:, :3], radii=spheres_np[:, 3]))
        rr.log("piper/gripper_spheres_tool_frame", rr.Points3D(positions=gripper_spheres[:, :3], radii=gripper_spheres[:, 3]))
        rr.log("piper/ee_frame", rr.Transform3D(translation=world_from_ee[:3, 3], mat3x3=world_from_ee[:3, :3]))
        rr.log("piper/tool_frame", rr.Transform3D(translation=world_from_tool[:3, 3], mat3x3=world_from_tool[:3, :3]))

    print(json.dumps({"save_dir": str(save_dir), "near_pair_count": len(near_pairs), "health": health}, indent=2))


def entrypoint() -> None:
    tyro.cli(piper_asset_debug)


if __name__ == "__main__":
    entrypoint()
