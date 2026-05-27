from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np


class PiperClientError(RuntimeError):
    """Raised when the Piper bridge rejects or fails a request."""


@dataclass(frozen=True)
class PiperRobotClient:
    """HTTP client for the TiPToP Piper runtime bridge."""

    base_url: str = "http://127.0.0.1:8766"
    timeout_s: float = 5.0
    execute_timeout_s: float = 120.0
    max_initial_error_rad: float = np.deg2rad(1.0)
    max_waypoint_step_rad: float = np.deg2rad(1.0)
    min_waypoint_dt_s: float = 0.5

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(path),
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s if timeout is None else timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise PiperClientError(f"Piper bridge request failed: {method} {path}: {exc}") from exc
        result = json.loads(body)
        if isinstance(result, dict) and result.get("success") is False:
            raise PiperClientError(str(result.get("error", result)))
        if isinstance(result, dict) and "error" in result:
            raise PiperClientError(str(result["error"]))
        return result

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

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

    def get_joint_positions(self) -> np.ndarray:
        result = self._request_json("GET", "/joint_positions")
        return np.asarray(result["positions"], dtype=np.float32)

    def get_gripper_width(self) -> float:
        result = self._request_json("GET", "/joint_positions")
        return float(result.get("gripper_width_m", 0.0))

    def get_snapshot(self) -> dict[str, np.ndarray]:
        try:
            with urllib.request.urlopen(self._url("/camera/snapshot.npz"), timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise PiperClientError(f"Piper bridge snapshot request failed: {exc}") from exc
        with np.load(io.BytesIO(raw)) as data:
            return {key: data[key] for key in data.files}

    def _validate_joint_path(self, joint_confs: np.ndarray, durations: np.ndarray) -> None:
        if joint_confs.ndim != 2 or joint_confs.shape[1] != 6:
            raise PiperClientError(f"Expected joint path shape (N, 6), got {joint_confs.shape}")
        if len(joint_confs) == 0:
            return
        if durations.ndim != 1 or len(durations) not in {1, len(joint_confs)}:
            raise PiperClientError("durations must have length 1 or match the number of waypoints")
        if np.any(durations < self.min_waypoint_dt_s):
            min_seen = float(np.min(durations))
            raise PiperClientError(
                f"Each waypoint duration must be >= {self.min_waypoint_dt_s:.3f}s; got {min_seen:.3f}s"
            )
        q_current = self.get_joint_positions()
        initial_error = float(np.max(np.abs(joint_confs[0] - q_current)))
        if initial_error > self.max_initial_error_rad:
            raise PiperClientError(
                f"First waypoint is {initial_error:.4f} rad from current joints; "
                f"limit is {self.max_initial_error_rad:.4f} rad"
            )
        if len(joint_confs) > 1:
            max_step = float(np.max(np.abs(np.diff(joint_confs, axis=0))))
            if max_step > self.max_waypoint_step_rad:
                raise PiperClientError(
                    f"Waypoint step is {max_step:.4f} rad; limit is {self.max_waypoint_step_rad:.4f} rad"
                )

    def execute_joint_impedance_path(self, joint_confs, joint_vels=None, durations=None):
        joint_confs = np.asarray(joint_confs, dtype=float)
        if durations is None:
            durations = np.full((len(joint_confs),), self.min_waypoint_dt_s, dtype=float)
        durations = np.asarray(durations, dtype=float).reshape(-1)
        durations = np.maximum(durations, self.min_waypoint_dt_s)
        self._validate_joint_path(joint_confs, durations)
        try:
            return self._request_json(
                "POST",
                "/execute_joint_path",
                {"positions": joint_confs.tolist(), "durations": durations.tolist()},
                timeout=self.execute_timeout_s,
            )
        except PiperClientError as exc:
            return {"success": False, "error": str(exc)}

    def open_gripper(self, speed: float = 0.05, force: float = 0.1):
        return self._gripper(width_m=0.07, force=force, speed=speed)

    def close_gripper(self, speed: float = 0.05, force: float = 0.1):
        return self._gripper(width_m=0.0, force=force, speed=speed)

    def set_enabled(self, enabled: bool):
        return self._request_json("POST", "/enable", {"enable": bool(enabled)})

    def _gripper(self, width_m: float, force: float, speed: float):
        try:
            return self._request_json("POST", "/gripper", {"width_m": width_m, "effort_n": force, "speed": speed})
        except PiperClientError as exc:
            return {"success": False, "error": str(exc)}


@cache
def get_piper_client() -> PiperRobotClient:
    return PiperRobotClient(base_url=os.environ.get("TIPTOP_PIPER_BRIDGE_URL", "http://127.0.0.1:8766"))
