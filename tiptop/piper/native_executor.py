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
        raise JointTrajectorySafetyError(f"duration {min_duration:.6f}s is shorter than {min_waypoint_dt_s:.6f}s")

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
