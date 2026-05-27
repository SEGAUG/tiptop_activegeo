from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import tyro
from curobo.types.base import TensorDeviceType
from cutamp.robots.piper import get_piper_kinematics_model

from tiptop.perception.cameras import get_hand_camera
from tiptop.piper import PiperRobotClient
from tiptop.scripts.calibrate_wrist_cam import (
    CHARUCOBOARD_CHECKER_SIZE,
    CHARUCOBOARD_MARKER_SIZE,
    CHARUCO_BOARD,
    SQUARES_X,
    SQUARES_Y,
    CharucoDetector,
)
from tiptop.piper.wrist_calibration_readiness import summarize_charuco_readiness


def _matrix_to_list(mat: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in mat]


def _fk_world_from_ee(q: np.ndarray) -> np.ndarray:
    tensor_args = TensorDeviceType()
    kin_model = get_piper_kinematics_model()
    q_t = tensor_args.to_device(q.astype(np.float32))[None]
    with torch.no_grad():
        return kin_model.get_state(q_t).ee_pose.get_numpy_matrix()[0]


def _capture_charuco_sample(save_dir: Path, name: str, client: PiperRobotClient) -> dict:
    cam = get_hand_camera(depth=True)
    try:
        frame = cam.read_camera()
        intrinsics = cam.get_intrinsics()
    finally:
        cam.close()

    q = client.get_joint_positions().astype(np.float64)
    world_from_ee = _fk_world_from_ee(q)
    detector = CharucoDetector(intrinsics)
    readings = detector.process_image(frame.bgr)
    viz = detector.augment_image(frame.serial, frame.bgr, visual_type=["markers", "charuco", "axes"])

    raw_path = save_dir / f"{name}_rgb.jpg"
    viz_path = save_dir / f"{name}_charuco_viz.jpg"
    cv2.imwrite(str(raw_path), frame.bgr)
    cv2.imwrite(str(viz_path), viz)

    payload = {
        "name": name,
        "camera_serial": frame.serial,
        "rgb_path": str(raw_path),
        "charuco_viz_path": str(viz_path),
        "detected": readings is not None,
        "aruco_markers": 0,
        "charuco_corners": 0,
        "q": [float(x) for x in q],
        "world_from_ee": _matrix_to_list(world_from_ee),
    }
    if readings is not None:
        corners, charuco_corners, charuco_ids, img_size = readings
        payload.update(
            {
                "aruco_markers": len(corners),
                "charuco_corners": len(charuco_corners),
                "charuco_id_min": int(charuco_ids.min()),
                "charuco_id_max": int(charuco_ids.max()),
                "image_size": [int(x) for x in img_size],
            }
        )
    return payload


def _require_safe_motion_health(health: dict) -> None:
    gates = health.get("control_gates", {})
    required = {
        "motion_allowed": health.get("motion_allowed") is True,
        "control_backend_sdk": health.get("control_backend") == "sdk",
        "enable_gate": gates.get("enable") is True,
        "joint_path_gate": gates.get("joint_path") is True,
        "sdk_gate": gates.get("sdk") is True,
    }
    missing = [name for name, ok in required.items() if not ok]
    if missing:
        raise RuntimeError(f"Remote bridge is not opened for real micro-motion: missing {missing}")


def piper_wrist_micro_calibration(
    bridge_url: str = "http://10.31.3.54:8766",
    output_dir: str = "/data/data2/jinhui.lin/code/aicode/piper_real_outputs/wrist_micro_calibration",
    action: Literal["status", "capture", "micro-joint", "micro-sequence"] = "capture",
    joint_index: int = 5,
    delta_rad: float = 0.003,
    sequence_delta_rad: float = 0.002,
    waypoint_dt_s: float = 5.0,
    min_feedback_motion_rad: float = 0.0005,
    confirm_real_robot_motion: bool = False,
) -> None:
    """Tiny wrist-camera calibration sampling helper for Piper.

    This is intentionally more conservative than the upstream `calibrate-wrist-cam`.
    It captures ChArUco samples from the wrist camera and, only with explicit
    confirmation plus remote motion gates, can execute one tiny joint step.
    """
    if abs(delta_rad) > np.deg2rad(0.5):
        raise ValueError("delta_rad must be <= 0.5 degrees for this micro-calibration helper")
    if abs(sequence_delta_rad) > np.deg2rad(0.5):
        raise ValueError("sequence_delta_rad must be <= 0.5 degrees for this micro-calibration helper")
    if waypoint_dt_s < 5.0:
        raise ValueError("waypoint_dt_s must be >= 5.0 seconds")

    client = PiperRobotClient(base_url=bridge_url, min_waypoint_dt_s=waypoint_dt_s, execute_timeout_s=60.0)
    health = client.health()
    q_before = client.get_joint_positions().astype(np.float64)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    save_dir = Path(output_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": timestamp,
        "bridge_url": bridge_url,
        "action": action,
        "confirm_real_robot_motion": confirm_real_robot_motion,
        "health_before": health,
        "charuco_board": {
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "checker_size_m": CHARUCOBOARD_CHECKER_SIZE,
            "marker_size_m": CHARUCOBOARD_MARKER_SIZE,
            "legacy_pattern": bool(CHARUCO_BOARD.getLegacyPattern()),
        },
        "q_before": [float(x) for x in q_before],
        "samples": [],
    }

    if action in {"capture", "micro-joint", "micro-sequence"}:
        payload["samples"].append(_capture_charuco_sample(save_dir, "sample_000_before", client))

    if action == "micro-joint":
        if joint_index < 0 or joint_index >= len(q_before):
            raise ValueError(f"joint_index must be in [0, {len(q_before) - 1}], got {joint_index}")
        q_target = q_before.copy()
        q_target[joint_index] += float(delta_rad)
        payload["q_target"] = [float(x) for x in q_target]
        payload["delta_rad"] = float(delta_rad)
        payload["waypoint_dt_s"] = float(waypoint_dt_s)

        if not confirm_real_robot_motion:
            payload["would_execute"] = True
            payload["message"] = "Dry-run only. Re-run with --confirm-real-robot-motion after onsite safety confirmation."
        else:
            _require_safe_motion_health(health)
            client.set_enabled(True)
            try:
                result = client.execute_joint_impedance_path(
                    np.stack([q_before, q_target], axis=0),
                    durations=np.array([waypoint_dt_s, waypoint_dt_s], dtype=float),
                )
            finally:
                client.set_enabled(False)
            payload["motion_result"] = result
            payload["health_after_motion"] = client.health()
            payload["q_after_motion"] = [float(x) for x in client.get_joint_positions()]
            payload["samples"].append(_capture_charuco_sample(save_dir, "sample_001_after", client))

    if action == "micro-sequence":
        if joint_index < 0 or joint_index >= len(q_before):
            raise ValueError(f"joint_index must be in [0, {len(q_before) - 1}], got {joint_index}")
        targets = []
        for offset in (sequence_delta_rad, -sequence_delta_rad, 0.0):
            q_target = q_before.copy()
            q_target[joint_index] += float(offset)
            targets.append(q_target)
        payload["sequence_delta_rad"] = float(sequence_delta_rad)
        payload["waypoint_dt_s"] = float(waypoint_dt_s)
        payload["min_feedback_motion_rad"] = float(min_feedback_motion_rad)
        payload["sequence_targets"] = [[float(x) for x in q] for q in targets]

        if not confirm_real_robot_motion:
            payload["would_execute"] = True
            payload["message"] = "Dry-run only. Re-run with --confirm-real-robot-motion after onsite safety confirmation."
        else:
            _require_safe_motion_health(health)
            payload["motion_results"] = []
            client.set_enabled(True)
            try:
                for sample_index, q_target in enumerate(targets, start=1):
                    q_current = client.get_joint_positions().astype(np.float64)
                    result = client.execute_joint_impedance_path(
                        np.stack([q_current, q_target], axis=0),
                        durations=np.array([waypoint_dt_s, waypoint_dt_s], dtype=float),
                    )
                    q_after_step = client.get_joint_positions().astype(np.float64)
                    max_feedback_delta = float(np.max(np.abs(q_after_step - q_current)))
                    result = {
                        **result,
                        "target_index": sample_index,
                        "q_before_step": [float(x) for x in q_current],
                        "q_target": [float(x) for x in q_target],
                        "q_after_step": [float(x) for x in q_after_step],
                        "max_feedback_delta_rad": max_feedback_delta,
                        "feedback_motion_detected": max_feedback_delta >= min_feedback_motion_rad,
                    }
                    payload["motion_results"].append(result)
                    if not result.get("success"):
                        break
                    payload["samples"].append(
                        _capture_charuco_sample(save_dir, f"sample_{sample_index:03d}_after_target", client)
                    )
            finally:
                client.set_enabled(False)
            payload["health_after_motion"] = client.health()
            payload["q_after_motion"] = [float(x) for x in client.get_joint_positions()]

    payload["readiness"] = summarize_charuco_readiness(payload)

    report_path = save_dir / "report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({"save_dir": str(save_dir), "report_path": str(report_path), **payload}, indent=2, ensure_ascii=False))


def entrypoint() -> None:
    tyro.cli(piper_wrist_micro_calibration)


if __name__ == "__main__":
    entrypoint()
