from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.policy.base_policy import InferencePolicy

from tiptop_molmospaces.action_queue import ActionQueue

log = logging.getLogger(__name__)


class TiPToPPolicy(InferencePolicy):
    """MolmoSpaces external policy adapter for modified TiPToP."""

    def __init__(self, exp_config: Any, task: Any | None = None):
        super().__init__(exp_config)
        self.exp_config = exp_config
        self.task = task
        self.policy_config = getattr(exp_config, "policy_config", exp_config)
        self.camera_names = list(getattr(self.policy_config, "camera_names", []))
        self.action_spec = dict(getattr(self.policy_config, "action_spec", {"arm": 7, "gripper": 1}))
        self.log_dir = Path(
            os.environ.get(
                "TIPTOP_MOLMOSPACES_LOG_DIR",
                str(getattr(self.policy_config, "log_dir", "logs")),
            )
        )
        self.action_queue = ActionQueue(
            action_spec=self.action_spec,
            default_gripper=float(getattr(self.policy_config, "safe_gripper_action", 255.0)),
        )
        self._schema_written = False
        self._observation_logged = False
        self._planning_exhausted = False
        self._hold_logged = False
        self._last_obs: Any | None = None
        self._last_arm = np.zeros(int(self.action_spec.get("arm", 7)), dtype=np.float32)
        self._last_gripper = np.array(
            [float(getattr(self.policy_config, "safe_gripper_action", 255.0))],
            dtype=np.float32,
        )
        self.prepare_model()

    def prepare_model(self, model_name: str | None = None) -> None:
        log.info("TiPToP MolmoSpaces adapter prepared; model_name=%s", model_name)
        if getattr(self.policy_config, "enable_tiptop_planning", False):
            import tiptop.perception_wrapper  # noqa: F401
            import tiptop.planning  # noqa: F401
            import tiptop.tiptop_run  # noqa: F401

            log.info("TiPToP planning modules imported")

    def reset(self) -> None:
        self.action_queue.clear()
        self._last_obs = None
        self._schema_written = False
        self._observation_logged = False
        self._planning_exhausted = False
        self._hold_logged = False
        log.info("TiPToPPolicy episode reset")

    def get_action(self, observation: Any) -> dict[str, np.ndarray]:
        obs = self._first_observation(observation)
        self._last_obs = obs
        self._log_observation(obs)
        self._write_schema_once(obs)

        queued = self.action_queue.pop()
        if queued is not None:
            log.info("Dequeued TiPToP action; remaining=%d", len(self.action_queue))
            return queued

        if not self._planning_exhausted:
            log.info("TiPToP planning start")
            try:
                plan = self.inference_model(self.obs_to_model_input(obs))
                if plan is not None:
                    self.action_queue.extend_plan(plan)
                    log.info("TiPToP planning success; queued action length=%d", len(self.action_queue))
                    queued = self.action_queue.pop()
                    if queued is not None:
                        return queued
                self._planning_exhausted = True
                log.info("TiPToP planning returned no executable plan")
            except Exception as exc:
                self._planning_exhausted = True
                log.exception("TiPToP planning failure; falling back to hold-current action: %s", exc)
            finally:
                log.info("TiPToP planning end")

        return self._hold_action(obs)

    def obs_to_model_input(self, observation: Any) -> Any:
        return observation

    def inference_model(self, model_input: Any) -> Any | None:
        serialized_plan_path = getattr(self.policy_config, "serialized_plan_path", None)
        if serialized_plan_path:
            path = Path(os.path.expandvars(os.path.expanduser(serialized_plan_path)))
            log.info("Loading serialized TiPToP plan from %s", path)
            with path.open("r") as f:
                return json.load(f)
        if not getattr(self.policy_config, "enable_tiptop_planning", False):
            return None
        raise NotImplementedError(
            "Live TiPToP planning requires configured perception/planning services; "
            "set serialized_plan_path for replay or extend inference_model for a live planner."
        )

    def model_output_to_action(self, model_output: Any) -> dict[str, np.ndarray]:
        if model_output is not None:
            self.action_queue.extend_plan(model_output)
        queued = self.action_queue.pop()
        if queued is not None:
            return queued
        return self._hold_action(self._last_obs)

    def _hold_action(self, observation: Any) -> dict[str, np.ndarray]:
        arm, gripper = self._extract_qpos(observation)
        self._last_arm = arm
        self._last_gripper = gripper
        if not self._hold_logged:
            log.info(
                "Returning hold-current joint-position action; arm=%s gripper=%s",
                arm.shape,
                gripper.shape,
            )
            self._hold_logged = True
        return {"arm": arm, "gripper": gripper}

    def _extract_qpos(self, observation: Any) -> tuple[np.ndarray, np.ndarray]:
        obs = self._first_observation(observation)
        qpos = None
        gripper = None

        if isinstance(obs, dict):
            if "robot_state" in obs and isinstance(obs["robot_state"], dict):
                qpos = obs["robot_state"].get("qpos", obs["robot_state"].get("joint_positions"))
                gripper = obs["robot_state"].get("gripper")
            if qpos is None and "qpos" in obs:
                if isinstance(obs["qpos"], dict):
                    qpos = obs["qpos"].get("arm")
                    gripper = obs["qpos"].get("gripper", gripper)
                else:
                    qpos = obs["qpos"]

        if qpos is None:
            robot_config = getattr(self.exp_config, "robot_config", None)
            qpos = getattr(robot_config, "init_qpos", None)
        if qpos is None:
            qpos = self._last_arm

        qpos_array = np.asarray(qpos, dtype=np.float32).reshape(-1)
        arm_size = int(self.action_spec.get("arm", 7))
        if qpos_array.size < arm_size:
            padded = np.zeros(arm_size, dtype=np.float32)
            padded[: qpos_array.size] = qpos_array
            qpos_array = padded
        arm = qpos_array[:arm_size].astype(np.float32, copy=False)

        if gripper is None and qpos_array.size > arm_size:
            gripper = qpos_array[arm_size]
        if gripper is None:
            gripper = self._last_gripper
        gripper_array = np.asarray(gripper, dtype=np.float32).reshape(-1)
        if gripper_array.size == 0:
            gripper_array = self._last_gripper
        gripper_array = np.array([float(gripper_array[0])], dtype=np.float32)
        return arm, gripper_array

    def _log_observation(self, observation: Any) -> None:
        obs = self._first_observation(observation)
        if self._observation_logged:
            return
        self._observation_logged = True
        if not isinstance(obs, dict):
            log.info("observation type=%s", type(obs).__name__)
            return
        log.info("observation keys=%s", sorted(obs.keys()))
        for name, image in self._iter_images(obs):
            shape = getattr(image, "shape", None)
            dtype = getattr(image, "dtype", None)
            log.info("camera %s image shape=%s dtype=%s", name, shape, dtype)
        arm, _gripper = self._extract_qpos(obs)
        log.info("robot_state qpos shape=%s", arm.shape)
        task_text = self._task_text(obs)
        if task_text:
            log.info("task/language visible=%s", task_text)

    def _write_schema_once(self, observation: Any) -> None:
        if self._schema_written:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "observation_schema.json"
        payload = self._schema(observation)
        payload["summary"] = self._observation_summary(observation)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self._schema_written = True
        log.info("Saved observation schema to %s", path)

    def _observation_summary(self, observation: Any) -> dict[str, Any]:
        obs = self._first_observation(observation)
        images = {}
        has_depth = False
        robot_state_keys: list[str] = []
        if isinstance(obs, dict):
            for name, image in self._iter_images(obs):
                images[name] = {
                    "shape": list(getattr(image, "shape", [])),
                    "dtype": str(getattr(image, "dtype", "")),
                }
            has_depth = any(key in obs for key in ("depth", "depths", "depth_images"))
            robot_state = obs.get("robot_state", obs.get("qpos"))
            if isinstance(robot_state, dict):
                robot_state_keys = sorted(robot_state.keys())
        arm, _ = self._extract_qpos(obs)
        return {
            "observation_type": type(obs).__name__,
            "top_level_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
            "camera_image_keys": sorted(images.keys()),
            "images": images,
            "has_depth": has_depth,
            "robot_state_keys": robot_state_keys,
            "qpos_shape": list(arm.shape),
            "task_or_language": self._task_text(obs),
        }

    def _schema(self, value: Any, depth: int = 0) -> dict[str, Any]:
        if depth > 5:
            return {"type": type(value).__name__}
        if isinstance(value, dict):
            return {
                "type": "dict",
                "keys": {str(k): self._schema(v, depth + 1) for k, v in value.items()},
            }
        if isinstance(value, list | tuple):
            preview = self._schema(value[0], depth + 1) if value else None
            return {"type": type(value).__name__, "length": len(value), "item_schema": preview}
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            return {"type": type(value).__name__, "shape": list(shape), "dtype": str(dtype)}
        return {"type": type(value).__name__}

    def _iter_images(self, observation: dict[str, Any]):
        image_containers = []
        for key in ("images", "image", "rgb", "camera_images"):
            if key in observation:
                image_containers.append(observation[key])
        for container in image_containers:
            if isinstance(container, dict):
                for name, image in container.items():
                    yield str(name), image
            else:
                yield "image", container
        for key, value in observation.items():
            if isinstance(key, str) and ("camera" in key or key in self.camera_names):
                if hasattr(value, "shape"):
                    yield key, value

    def _task_text(self, observation: Any) -> str | None:
        if self.task is not None and hasattr(self.task, "get_task_description"):
            try:
                return str(self.task.get_task_description())
            except Exception:
                pass
        if isinstance(observation, dict):
            for key in ("task", "text", "instruction", "language", "task_description"):
                value = observation.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    for nested in ("task_description", "text", "instruction"):
                        if isinstance(value.get(nested), str):
                            return value[nested]
        return None

    def _first_observation(self, observation: Any) -> Any:
        if isinstance(observation, list | tuple) and observation:
            return observation[0]
        return observation
