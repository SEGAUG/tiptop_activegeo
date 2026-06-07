from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class ActionQueue:
    """FIFO queue that converts TiPToP serialized plans into MolmoSpaces actions."""

    def __init__(self, action_spec: dict[str, int] | None = None, default_gripper: float = 255.0):
        self.action_spec = action_spec or {"arm": 7, "gripper": 1}
        self.default_gripper = float(default_gripper)
        self._queue: deque[dict[str, np.ndarray]] = deque()
        self._last_gripper = self.default_gripper

    def __len__(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()
        self._last_gripper = self.default_gripper

    def pop(self) -> dict[str, np.ndarray] | None:
        if not self._queue:
            return None
        action = self._queue.popleft()
        self._last_gripper = float(action["gripper"][0])
        return action

    def push_action(self, arm: Any, gripper: Any | None = None) -> None:
        self._queue.append(self._normalize_action(arm, gripper))

    def extend_plan(self, plan: Any) -> None:
        if plan is None:
            return

        steps = plan.get("steps", plan) if isinstance(plan, dict) else plan
        if not isinstance(steps, list | tuple):
            raise TypeError(f"Unsupported TiPToP plan type: {type(plan).__name__}")

        for step in steps:
            if isinstance(step, dict):
                step_type = str(step.get("type", "")).lower()
                if step_type in {"trajectory", "joint_trajectory", "path"}:
                    positions = step.get("positions", step.get("q", step.get("trajectory")))
                    self._extend_positions(positions, step.get("gripper"))
                elif step_type in {"gripper", "grasp"}:
                    self.push_action(self._last_arm_or_zeros(), step.get("action", step.get("value")))
                elif "arm" in step or "qpos" in step or "positions" in step:
                    self.push_action(step.get("arm", step.get("qpos", step.get("positions"))), step.get("gripper"))
                else:
                    raise ValueError(f"Unsupported plan step keys: {sorted(step.keys())}")
            else:
                self.push_action(step, None)

    def _extend_positions(self, positions: Any, gripper: Any | None = None) -> None:
        array = np.asarray(positions, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        for row in array:
            self.push_action(row, gripper)

    def _normalize_action(self, arm: Any, gripper: Any | None = None) -> dict[str, np.ndarray]:
        arm_array = np.asarray(arm, dtype=np.float32).reshape(-1)
        arm_size = int(self.action_spec.get("arm", 7))
        if arm_array.size < arm_size:
            raise ValueError(f"Arm action has {arm_array.size} values, expected at least {arm_size}")
        arm_array = arm_array[:arm_size].astype(np.float32, copy=False)

        if gripper is None and np.asarray(arm, dtype=np.float32).reshape(-1).size > arm_size:
            gripper = np.asarray(arm, dtype=np.float32).reshape(-1)[arm_size]
        gripper_array = np.array([self._normalize_gripper(gripper)], dtype=np.float32)
        self._last_gripper = float(gripper_array[0])
        return {"arm": arm_array, "gripper": gripper_array}

    def _normalize_gripper(self, value: Any | None) -> float:
        if value is None:
            return self._last_gripper
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"close", "closed", "grasp"}:
                return 0.0
            if lowered in {"open", "opened", "release"}:
                return 255.0
            raise ValueError(f"Unknown gripper action string: {value}")
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return self._last_gripper
        return float(array[0])

    def _last_arm_or_zeros(self) -> np.ndarray:
        if self._queue:
            return self._queue[-1]["arm"]
        return np.zeros(int(self.action_spec.get("arm", 7)), dtype=np.float32)
