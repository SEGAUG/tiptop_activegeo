# TiPToP Piper Native Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the left Piper a TiPToP-native single-arm robot path with wrist-camera calibration validation, Piper cuRobo asset validation, planning dry-run, and gated slow execution.

**Architecture:** TiPToP owns perception, IK, collision checking, and trajectory generation. The Piper bridge is only a low-level transport for already-planned 6-DOF joint trajectories and gripper commands. Every real-motion path remains gated, slow, and dry-run by default.

**Tech Stack:** Python 3.12, pytest, tyro CLI, NumPy, cuRobo/cuTAMP, Rerun, OpenCV ChArUco, existing HTTP Piper bridge.

---

## File Structure

- `tiptop/piper/piper_client.py`: HTTP client and client-side safety checks for transport-only execution.
- `tiptop/piper/native_executor.py`: new pure-Python trajectory report and gate logic used before any low-level execution.
- `tiptop/piper/wrist_calibration_readiness.py`: new pure-Python helper that summarizes ChArUco captures and decides whether calibration data is ready.
- `tiptop/piper/asset_validation.py`: new pure-Python report checks for Piper asset debug outputs.
- `tiptop/scripts/piper_wrist_micro_calibration.py`: add readiness summary to each capture report.
- `tiptop/scripts/piper_asset_debug.py`: add pass/fail asset validation fields to report output.
- `tiptop/scripts/piper_native_execution_dry_run.py`: new CLI that validates a planned joint trajectory without moving the robot.
- `tiptop/scripts/piper_planning_dry_run.py`: new CLI that builds Piper cuRobo solvers and saves a no-execution planning report.
- `tiptop/workspace.py`: replace empty Piper workspace with conservative cuboids for the current desk setup.
- `pyproject.toml`: register new scripts.
- `tests/test_piper_client.py`: extend client safety tests.
- `tests/test_piper_native_executor.py`: new unit tests for trajectory report and gate behavior.
- `tests/test_piper_wrist_calibration_readiness.py`: new unit tests for ChArUco readiness decisions.
- `tests/test_piper_asset_validation.py`: new unit tests for asset validation decisions.
- `tests/test_piper_workspace.py`: new unit tests for conservative workspace cuboids.
- `piper_real_robot_runbook.md`: update commands for the B方案 native TiPToP path.

## Task 1: Strengthen Piper Client Transport Safety

**Files:**
- Modify: `tiptop/piper/piper_client.py`
- Modify: `tests/test_piper_client.py`

- [ ] **Step 1: Add failing tests for duration and bridge gate validation**

Append these tests to `tests/test_piper_client.py`:

```python
def test_validate_joint_path_rejects_too_short_duration():
    client = StaticPiperClient(min_waypoint_dt_s=1.0)
    with pytest.raises(PiperClientError, match="duration"):
        client._validate_joint_path(np.zeros((2, 6)), np.array([0.5, 1.0]))


def test_validate_health_for_motion_requires_all_gates():
    client = StaticPiperClient()
    health = {
        "motion_allowed": True,
        "control_backend": "sdk",
        "control_gates": {"enable": True, "joint_path": True, "sdk": False},
    }
    with pytest.raises(PiperClientError, match="sdk"):
        client.validate_health_for_motion(health)


def test_validate_health_for_motion_accepts_open_sdk_bridge():
    client = StaticPiperClient()
    health = {
        "motion_allowed": True,
        "control_backend": "sdk",
        "control_gates": {"enable": True, "joint_path": True, "sdk": True},
    }
    client.validate_health_for_motion(health)
```

- [ ] **Step 2: Run tests and verify the new failures**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_client.py -v
```

Expected: failures for missing `validate_health_for_motion` and duration rejection.

- [ ] **Step 3: Implement the client validation methods**

Add this method inside `PiperRobotClient` in `tiptop/piper/piper_client.py`:

```python
    def validate_health_for_motion(self, health: dict[str, Any] | None = None) -> None:
        if health is None:
            health = self.health()
        gates = health.get("control_gates", {})
        missing = []
        if health.get("motion_allowed") is not True:
            missing.append("motion_allowed")
        if health.get("control_backend") != "sdk":
            missing.append("control_backend=sdk")
        for gate_name in ("enable", "joint_path", "sdk"):
            if gates.get(gate_name) is not True:
                missing.append(gate_name)
        if missing:
            raise PiperClientError(f"Piper bridge is not open for real motion: missing {missing}")
```

Update `_validate_joint_path` in the same file after the existing `durations` shape check:

```python
        if np.any(durations < self.min_waypoint_dt_s):
            min_seen = float(np.min(durations))
            raise PiperClientError(
                f"Each waypoint duration must be >= {self.min_waypoint_dt_s:.3f}s; got {min_seen:.3f}s"
            )
```

- [ ] **Step 4: Run unit tests**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_client.py -v
```

Expected: all `tests/test_piper_client.py` tests pass.

- [ ] **Step 5: Commit**

```bash
git add tiptop/piper/piper_client.py tests/test_piper_client.py
git commit -m "test: strengthen Piper transport safety checks"
```

## Task 2: Add Native Executor Dry-Run Report

**Files:**
- Create: `tiptop/piper/native_executor.py`
- Create: `tests/test_piper_native_executor.py`
- Create: `tiptop/scripts/piper_native_execution_dry_run.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing tests for trajectory safety report**

Create `tests/test_piper_native_executor.py`:

```python
import numpy as np
import pytest

from tiptop.piper.native_executor import JointTrajectorySafetyError, build_joint_trajectory_report


def test_build_joint_trajectory_report_accepts_slow_small_path():
    q_current = np.zeros(6)
    positions = np.array([[0, 0, 0, 0, 0, 0], [0.01, 0, 0, 0, 0, 0]], dtype=float)
    report = build_joint_trajectory_report(q_current, positions, durations=[5.0, 5.0])
    assert report["safe_for_real_motion"] is True
    assert report["waypoint_count"] == 2
    assert report["max_step_rad"] == pytest.approx(0.01)
    assert report["min_duration_s"] == pytest.approx(5.0)


def test_build_joint_trajectory_report_rejects_wrong_shape():
    with pytest.raises(JointTrajectorySafetyError, match="shape"):
        build_joint_trajectory_report(np.zeros(6), np.zeros((2, 7)), durations=[5.0, 5.0])


def test_build_joint_trajectory_report_rejects_large_initial_error():
    with pytest.raises(JointTrajectorySafetyError, match="initial"):
        build_joint_trajectory_report(np.zeros(6), np.ones((2, 6)) * 0.1, durations=[5.0, 5.0])


def test_build_joint_trajectory_report_rejects_fast_waypoint():
    positions = np.zeros((2, 6))
    with pytest.raises(JointTrajectorySafetyError, match="duration"):
        build_joint_trajectory_report(np.zeros(6), positions, durations=[5.0, 1.0], min_waypoint_dt_s=3.0)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_native_executor.py -v
```

Expected: import failure for `tiptop.piper.native_executor`.

- [ ] **Step 3: Implement the pure safety report helper**

Create `tiptop/piper/native_executor.py`:

```python
from __future__ import annotations

from typing import Sequence

import numpy as np


class JointTrajectorySafetyError(RuntimeError):
    """Raised when a Piper joint trajectory is unsafe to submit."""


def build_joint_trajectory_report(
    q_current: np.ndarray,
    positions: np.ndarray,
    durations: Sequence[float],
    *,
    max_initial_error_rad: float = np.deg2rad(1.0),
    max_waypoint_step_rad: float = np.deg2rad(1.0),
    min_waypoint_dt_s: float = 3.0,
) -> dict:
    q_current = np.asarray(q_current, dtype=float).reshape(-1)
    positions = np.asarray(positions, dtype=float)
    durations = np.asarray(list(durations), dtype=float).reshape(-1)

    if q_current.shape != (6,):
        raise JointTrajectorySafetyError(f"q_current must have shape (6,), got {q_current.shape}")
    if positions.ndim != 2 or positions.shape[1] != 6:
        raise JointTrajectorySafetyError(f"positions must have shape (N, 6), got {positions.shape}")
    if positions.shape[0] == 0:
        raise JointTrajectorySafetyError("positions must contain at least one waypoint")
    if len(durations) not in {1, positions.shape[0]}:
        raise JointTrajectorySafetyError("durations must have length 1 or match waypoint count")
    if len(durations) == 1:
        durations = np.full((positions.shape[0],), float(durations[0]), dtype=float)

    initial_error = float(np.max(np.abs(positions[0] - q_current)))
    if initial_error > max_initial_error_rad:
        raise JointTrajectorySafetyError(
            f"initial waypoint error {initial_error:.6f} rad exceeds {max_initial_error_rad:.6f} rad"
        )

    max_step = 0.0
    if positions.shape[0] > 1:
        max_step = float(np.max(np.abs(np.diff(positions, axis=0))))
        if max_step > max_waypoint_step_rad:
            raise JointTrajectorySafetyError(
                f"waypoint step {max_step:.6f} rad exceeds {max_waypoint_step_rad:.6f} rad"
            )

    min_duration = float(np.min(durations))
    if min_duration < min_waypoint_dt_s:
        raise JointTrajectorySafetyError(
            f"duration {min_duration:.6f}s is shorter than {min_waypoint_dt_s:.6f}s"
        )

    return {
        "safe_for_real_motion": True,
        "waypoint_count": int(positions.shape[0]),
        "duration_count": int(len(durations)),
        "total_duration_s": float(np.sum(durations)),
        "initial_error_rad": initial_error,
        "max_step_rad": max_step,
        "min_duration_s": min_duration,
        "max_initial_error_rad": float(max_initial_error_rad),
        "max_waypoint_step_rad": float(max_waypoint_step_rad),
        "min_waypoint_dt_s": float(min_waypoint_dt_s),
    }
```

- [ ] **Step 4: Add the dry-run CLI**

Create `tiptop/scripts/piper_native_execution_dry_run.py`:

```python
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
```

- [ ] **Step 5: Register the script**

Add this line under Piper scripts in `pyproject.toml`:

```toml
piper-native-execution-dry-run = "tiptop.scripts.piper_native_execution_dry_run:entrypoint"
```

- [ ] **Step 6: Run tests**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_native_executor.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add tiptop/piper/native_executor.py tiptop/scripts/piper_native_execution_dry_run.py tests/test_piper_native_executor.py pyproject.toml
git commit -m "feat: add Piper native execution dry-run report"
```

## Task 3: Add Wrist Calibration Readiness Summary

**Files:**
- Create: `tiptop/piper/wrist_calibration_readiness.py`
- Create: `tests/test_piper_wrist_calibration_readiness.py`
- Modify: `tiptop/scripts/piper_wrist_micro_calibration.py`

- [ ] **Step 1: Write failing readiness tests**

Create `tests/test_piper_wrist_calibration_readiness.py`:

```python
from tiptop.piper.wrist_calibration_readiness import summarize_charuco_readiness


def test_summarize_charuco_readiness_accepts_good_sample():
    payload = {
        "samples": [
            {"detected": True, "charuco_corners": 70, "aruco_markers": 44, "camera_serial": "243722072079"}
        ]
    }
    summary = summarize_charuco_readiness(payload, min_corners=50, min_markers=20)
    assert summary["ready"] is True
    assert summary["best_charuco_corners"] == 70
    assert summary["sample_count"] == 1


def test_summarize_charuco_readiness_rejects_missing_board():
    payload = {"samples": [{"detected": False, "charuco_corners": 0, "aruco_markers": 0}]}
    summary = summarize_charuco_readiness(payload, min_corners=50, min_markers=20)
    assert summary["ready"] is False
    assert "not enough ChArUco corners" in summary["reason"]


def test_summarize_charuco_readiness_counts_detected_samples():
    payload = {
        "samples": [
            {"detected": False, "charuco_corners": 0, "aruco_markers": 0},
            {"detected": True, "charuco_corners": 55, "aruco_markers": 22},
        ]
    }
    summary = summarize_charuco_readiness(payload, min_corners=50, min_markers=20)
    assert summary["detected_sample_count"] == 1
    assert summary["ready"] is True
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_wrist_calibration_readiness.py -v
```

Expected: import failure for `tiptop.piper.wrist_calibration_readiness`.

- [ ] **Step 3: Implement the readiness helper**

Create `tiptop/piper/wrist_calibration_readiness.py`:

```python
from __future__ import annotations


def summarize_charuco_readiness(payload: dict, *, min_corners: int = 50, min_markers: int = 20) -> dict:
    samples = list(payload.get("samples", []))
    detected = [s for s in samples if s.get("detected") is True]
    best_corners = max([int(s.get("charuco_corners", 0)) for s in samples], default=0)
    best_markers = max([int(s.get("aruco_markers", 0)) for s in samples], default=0)

    ready = best_corners >= min_corners and best_markers >= min_markers
    if ready:
        reason = "ChArUco detection is sufficient for calibration sampling"
    elif best_corners < min_corners:
        reason = f"not enough ChArUco corners: {best_corners} < {min_corners}"
    else:
        reason = f"not enough ArUco markers: {best_markers} < {min_markers}"

    return {
        "ready": ready,
        "reason": reason,
        "sample_count": len(samples),
        "detected_sample_count": len(detected),
        "best_charuco_corners": best_corners,
        "best_aruco_markers": best_markers,
        "min_corners": int(min_corners),
        "min_markers": int(min_markers),
    }
```

- [ ] **Step 4: Add readiness summary to capture reports**

In `tiptop/scripts/piper_wrist_micro_calibration.py`, add:

```python
from tiptop.piper.wrist_calibration_readiness import summarize_charuco_readiness
```

Before writing `report_path`, add:

```python
    payload["readiness"] = summarize_charuco_readiness(payload)
```

- [ ] **Step 5: Run tests**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_wrist_calibration_readiness.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run a read-only capture on the real bridge**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
  pixi run piper-wrist-micro-calibration \
  --bridge-url http://10.31.3.54:8766 \
  --action capture
```

Expected: JSON output contains `"readiness": {"ready": true, ...}` when the board remains visible.

- [ ] **Step 7: Commit**

```bash
git add tiptop/piper/wrist_calibration_readiness.py tiptop/scripts/piper_wrist_micro_calibration.py tests/test_piper_wrist_calibration_readiness.py
git commit -m "feat: report Piper wrist calibration readiness"
```

## Task 4: Add Piper Asset Validation Report

**Files:**
- Create: `tiptop/piper/asset_validation.py`
- Create: `tests/test_piper_asset_validation.py`
- Modify: `tiptop/scripts/piper_asset_debug.py`

- [ ] **Step 1: Write failing asset validation tests**

Create `tests/test_piper_asset_validation.py`:

```python
from tiptop.piper.asset_validation import validate_asset_debug_payload


def test_validate_asset_debug_payload_accepts_clear_asset():
    payload = {
        "link_sphere_count": 42,
        "gripper_sphere_count": 6,
        "near_pairs": [{"margin_m": 0.012}, {"margin_m": 0.02}],
    }
    report = validate_asset_debug_payload(payload, min_link_spheres=20, min_gripper_spheres=4)
    assert report["asset_ready_for_planning_dry_run"] is True
    assert report["min_self_collision_margin_m"] == 0.012


def test_validate_asset_debug_payload_rejects_missing_gripper_spheres():
    payload = {"link_sphere_count": 42, "gripper_sphere_count": 0, "near_pairs": []}
    report = validate_asset_debug_payload(payload, min_link_spheres=20, min_gripper_spheres=4)
    assert report["asset_ready_for_planning_dry_run"] is False
    assert "gripper_sphere_count" in report["reasons"][0]


def test_validate_asset_debug_payload_rejects_intersecting_spheres():
    payload = {"link_sphere_count": 42, "gripper_sphere_count": 6, "near_pairs": [{"margin_m": -0.001}]}
    report = validate_asset_debug_payload(payload, min_link_spheres=20, min_gripper_spheres=4)
    assert report["asset_ready_for_planning_dry_run"] is False
    assert "self-collision" in " ".join(report["reasons"])
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_asset_validation.py -v
```

Expected: import failure for `tiptop.piper.asset_validation`.

- [ ] **Step 3: Implement the validator**

Create `tiptop/piper/asset_validation.py`:

```python
from __future__ import annotations


def validate_asset_debug_payload(
    payload: dict,
    *,
    min_link_spheres: int = 20,
    min_gripper_spheres: int = 4,
    min_allowed_margin_m: float = 0.0,
) -> dict:
    reasons: list[str] = []
    link_count = int(payload.get("link_sphere_count", 0))
    gripper_count = int(payload.get("gripper_sphere_count", 0))
    near_pairs = list(payload.get("near_pairs", []))
    margins = [float(row.get("margin_m", 0.0)) for row in near_pairs]
    min_margin = min(margins) if margins else None

    if link_count < min_link_spheres:
        reasons.append(f"link_sphere_count {link_count} < {min_link_spheres}")
    if gripper_count < min_gripper_spheres:
        reasons.append(f"gripper_sphere_count {gripper_count} < {min_gripper_spheres}")
    if min_margin is not None and min_margin < min_allowed_margin_m:
        reasons.append(f"near self-collision margin {min_margin:.6f} m < {min_allowed_margin_m:.6f} m")

    return {
        "asset_ready_for_planning_dry_run": len(reasons) == 0,
        "reasons": reasons,
        "link_sphere_count": link_count,
        "gripper_sphere_count": gripper_count,
        "near_pair_count": len(near_pairs),
        "min_self_collision_margin_m": min_margin,
        "min_link_spheres": int(min_link_spheres),
        "min_gripper_spheres": int(min_gripper_spheres),
        "min_allowed_margin_m": float(min_allowed_margin_m),
    }
```

- [ ] **Step 4: Add validation output to asset debug**

In `tiptop/scripts/piper_asset_debug.py`, add:

```python
from tiptop.piper.asset_validation import validate_asset_debug_payload
```

After `payload` is assembled and before writing files, add:

```python
    payload["validation"] = validate_asset_debug_payload(payload)
```

In `_write_markdown_report`, add these lines after the gripper sphere count line:

```python
        f"- asset_ready_for_planning_dry_run: `{payload['validation']['asset_ready_for_planning_dry_run']}`",
        f"- validation_reasons: `{payload['validation']['reasons']}`",
```

- [ ] **Step 5: Run tests**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_asset_validation.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run read-only asset debug**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
  pixi run piper-asset-debug \
  --bridge-url http://10.31.3.54:8766 \
  --no-rr-spawn
```

Expected: command refuses to run if `motion_allowed=true`; otherwise it writes `report.json`, `report.md`, and validation fields.

- [ ] **Step 7: Commit**

```bash
git add tiptop/piper/asset_validation.py tiptop/scripts/piper_asset_debug.py tests/test_piper_asset_validation.py
git commit -m "feat: validate Piper cuRobo asset reports"
```

## Task 5: Add Conservative Piper Workspace Cuboids

**Files:**
- Modify: `tiptop/workspace.py`
- Create: `tests/test_piper_workspace.py`

- [ ] **Step 1: Write failing workspace tests**

Create `tests/test_piper_workspace.py`:

```python
from tiptop.workspace import piper_workspace


def test_piper_workspace_contains_required_obstacles():
    names = {cuboid.name for cuboid in piper_workspace()}
    assert {"desk", "rear_glass", "right_arm_keepout", "laptop_keepout"}.issubset(names)


def test_piper_workspace_obstacles_have_positive_dimensions():
    for cuboid in piper_workspace():
        assert all(dim > 0 for dim in cuboid.dims)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_workspace.py -v
```

Expected: failure because `piper_workspace()` is currently empty.

- [ ] **Step 3: Implement conservative workspace cuboids**

Replace `piper_workspace()` in `tiptop/workspace.py` with:

```python
def piper_workspace() -> tuple[Cuboid, ...]:
    desk = Cuboid(
        "desk",
        dims=[1.20, 0.90, 0.04],
        pose=[0.35, 0.00, -0.04, *unit_quat],
        color=[235, 235, 235],
    )
    rear_glass = Cuboid(
        "rear_glass",
        dims=[1.20, 0.04, 0.55],
        pose=[0.45, 0.46, 0.28, *unit_quat],
        color=[120, 200, 255],
    )
    right_arm_keepout = Cuboid(
        "right_arm_keepout",
        dims=[0.45, 0.35, 0.55],
        pose=[0.20, -0.42, 0.25, *unit_quat],
        color=[255, 80, 80],
    )
    laptop_keepout = Cuboid(
        "laptop_keepout",
        dims=[0.42, 0.30, 0.08],
        pose=[0.45, -0.35, 0.04, *unit_quat],
        color=[40, 40, 40],
    )
    ceiling = Cuboid(
        "ceiling",
        dims=[1.30, 1.10, 0.03],
        pose=[0.35, 0.00, 0.75, *unit_quat],
        color=[255, 255, 255],
    )
    return (desk, rear_glass, right_arm_keepout, laptop_keepout, ceiling)
```

- [ ] **Step 4: Run workspace tests**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest tests/test_piper_workspace.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Generate workspace visualization**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run python tiptop/workspace.py
```

Expected: Rerun opens or saves with `workspace_desk`, `workspace_rear_glass`, `workspace_right_arm_keepout`, `workspace_laptop_keepout`, and `workspace_ceiling`.

- [ ] **Step 6: Commit**

```bash
git add tiptop/workspace.py tests/test_piper_workspace.py
git commit -m "feat: add conservative Piper workspace"
```

## Task 6: Add Piper Planning Dry-Run CLI

**Files:**
- Create: `tiptop/scripts/piper_planning_dry_run.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create a dry-run planning script**

Create `tiptop/scripts/piper_planning_dry_run.py`:

```python
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
```

- [ ] **Step 2: Register the script**

Add this line in `pyproject.toml`:

```toml
piper-planning-dry-run = "tiptop.scripts.piper_planning_dry_run:entrypoint"
```

- [ ] **Step 3: Run import smoke**

Run:

```bash
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run python -c "from tiptop.scripts.piper_planning_dry_run import piper_planning_dry_run; print(piper_planning_dry_run.__name__)"
```

Expected output:

```text
piper_planning_dry_run
```

- [ ] **Step 4: Run read-only planning dry-run**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
  pixi run piper-planning-dry-run \
  --bridge-url http://10.31.3.54:8766 \
  --joint-index 5 \
  --delta-rad 0.005 \
  --waypoint-dt-s 5.0
```

Expected: a `trajectory.npz` and `report.json` are saved; `executed=false`; bridge motion gate remains false.

- [ ] **Step 5: Validate the saved trajectory with native dry-run CLI**

Find the newest trajectory from Step 4 and validate it:

```bash
LATEST_TRAJ="$(find /data/data2/jinhui.lin/code/aicode/piper_real_outputs/planning_dry_run -name trajectory.npz -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$LATEST_TRAJ"
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
  pixi run piper-native-execution-dry-run \
  --bridge-url http://10.31.3.54:8766 \
  --trajectory-npz "$LATEST_TRAJ"
```

Expected: safety report says `"safe_for_real_motion": true`; no robot motion.

- [ ] **Step 6: Commit**

```bash
git add tiptop/scripts/piper_planning_dry_run.py pyproject.toml
git commit -m "feat: add Piper cuRobo planning dry-run"
```

## Task 7: Update Runbook for B方案 Native Path

**Files:**
- Modify: `../piper_real_robot_runbook.md`

- [ ] **Step 1: Add the native path section**

Append this section to `/data/data2/jinhui.lin/code/aicode/piper_real_robot_runbook.md`:

````markdown
## 2026-05-27 B方案：TiPToP Piper 原生执行路径

当前决策：不把 pika 作为运动系统。pika 只作为参考；TiPToP/cuRobo 负责 IK、轨迹和碰撞，bridge 只负责低层 Piper transport。

推荐验证顺序：

1. `piper-wrist-micro-calibration --action capture`
2. `piper-asset-debug`
3. `piper-planning-dry-run`
4. `piper-native-execution-dry-run`
5. 现场确认后再做极慢真实 smoke test

所有命令默认要求 `motion_allowed=false`。如果发现 bridge 已打开 motion gate，先恢复只读 bridge，再继续验证。

只读健康检查：

```bash
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
curl -sS --max-time 5 http://10.31.3.54:8766/health | python3 -m json.tool
```

腕部相机只读检查：

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
pixi run piper-wrist-micro-calibration \
  --bridge-url http://10.31.3.54:8766 \
  --action capture
```

Piper asset 只读检查：

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
pixi run piper-asset-debug \
  --bridge-url http://10.31.3.54:8766 \
  --no-rr-spawn
```
````

- [ ] **Step 2: Verify markdown section was added**

Run:

```bash
rg -n "B方案|piper-planning-dry-run|piper-native-execution-dry-run" /data/data2/jinhui.lin/code/aicode/piper_real_robot_runbook.md
```

Expected: all three patterns are found.

- [ ] **Step 3: Commit**

```bash
git add /data/data2/jinhui.lin/code/aicode/piper_real_robot_runbook.md
git commit -m "docs: update Piper native execution runbook"
```

## Task 8: Final Verification Bundle

**Files:**
- Read: all files changed in Tasks 1-7

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
  pixi run pytest \
  tests/test_piper_client.py \
  tests/test_piper_native_executor.py \
  tests/test_piper_wrist_calibration_readiness.py \
  tests/test_piper_asset_validation.py \
  tests/test_piper_workspace.py \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Verify real bridge remains read-only**

Run:

```bash
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
curl -sS --max-time 5 http://10.31.3.54:8766/health | python3 -m json.tool
```

Expected: `"motion_allowed": false`, `"control_gates": {"enable": false, "joint_path": false, "sdk": false}`.

- [ ] **Step 3: Run read-only wrist capture**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
  pixi run piper-wrist-micro-calibration \
  --bridge-url http://10.31.3.54:8766 \
  --action capture
```

Expected: `readiness.ready=true` if ChArUco board is still visible.

- [ ] **Step 4: Run read-only asset debug**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
  pixi run piper-asset-debug \
  --bridge-url http://10.31.3.54:8766 \
  --no-rr-spawn
```

Expected: report contains `validation.asset_ready_for_planning_dry_run`.

- [ ] **Step 5: Run planning dry-run**

Run:

```bash
cd /data/data2/jinhui.lin/code/aicode/tiptop
PATH="$HOME/.pixi/bin:$PATH" PYTHONNOUSERSITE=1 \
NO_PROXY=10.31.3.54,127.0.0.1,localhost \
no_proxy=10.31.3.54,127.0.0.1,localhost \
  pixi run piper-planning-dry-run \
  --bridge-url http://10.31.3.54:8766 \
  --joint-index 5 \
  --delta-rad 0.005 \
  --waypoint-dt-s 5.0
```

Expected: a no-execution report and trajectory file are saved.

- [ ] **Step 6: Commit final verification notes if runbook changed**

If any final output paths are added to `piper_real_robot_runbook.md`, run:

```bash
git add /data/data2/jinhui.lin/code/aicode/piper_real_robot_runbook.md
git commit -m "docs: record Piper native verification outputs"
```
