from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from molmo_spaces.policy.base_policy import InferencePolicy

from tiptop_molmospaces.action_queue import ActionQueue

log = logging.getLogger(__name__)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
        self.live_planning_enabled = _env_enabled(
            "TIPTOP_ENABLE_LIVE_PLANNING",
            bool(getattr(self.policy_config, "enable_tiptop_planning", False)),
        )
        self.require_plan = _env_enabled("TIPTOP_REQUIRE_PLAN", False)
        self.backend = os.environ.get("TIPTOP_VLM_BACKEND", "qwen" if self.live_planning_enabled else "disabled")
        self.live_log_dir = self.log_dir / "molmospaces_live"
        self._episode_counter = -1
        self._episode_id = "episode_000000"
        self._episode_dir = self.live_log_dir / self._episode_id
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
        self._episode_counter += 1
        self._episode_id = f"episode_{self._episode_counter:06d}"
        self._episode_dir = self.live_log_dir / self._episode_id
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
                if self.live_planning_enabled:
                    plan_result = self.plan_from_observation(obs)
                    plan = self._plan_result_to_queue(plan_result, obs)
                    self._record_plan_result_metrics(plan_result, obs)
                    if not plan_result.get("success", False):
                        self._write_planning_failure(plan_result)
                else:
                    plan = self.inference_model(self.obs_to_model_input(obs))
                if plan is not None:
                    if not self.live_planning_enabled:
                        self.action_queue.extend_plan(plan)
                    log.info("TiPToP planning success; queued action length=%d", len(self.action_queue))
                    queued = self.action_queue.pop()
                    self._planning_exhausted = True
                    if queued is not None:
                        return queued
                self._planning_exhausted = True
                log.info("TiPToP planning returned no executable plan")
            except Exception as exc:
                self._planning_exhausted = True
                if self.live_planning_enabled:
                    plan_result = {
                        "success": False,
                        "failure_reason": str(exc),
                        "debug": {"failure_layer": "exception", "exception_type": type(exc).__name__},
                    }
                    self._write_planning_failure(plan_result)
                    self._record_plan_result_metrics(plan_result, obs)
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

    def plan_from_observation(self, observation: Any) -> dict[str, Any]:
        """Attempt live planning from MolmoSpaces policy observations.

        The current MolmoSpaces policy observation exposes RGB cameras and robot qpos,
        but the smoke schema does not expose depth. We record that as an explicit
        planning blocker instead of silently using privileged simulator state.
        """
        obs = self._first_observation(observation)
        self._episode_dir.mkdir(parents=True, exist_ok=True)

        images = dict(self._iter_images(obs)) if isinstance(obs, dict) else {}
        depth_maps = self._extract_depth_maps(obs)
        intrinsics = self._extract_camera_intrinsics(obs)
        arm, gripper = self._extract_qpos(obs)
        task_text = self._task_text(obs)

        for name, image in images.items():
            self._save_rgb_debug(name, image)
        for name, depth in depth_maps.items():
            np.save(self._episode_dir / f"depth_{self._safe_name(name)}.npy", np.asarray(depth))

        summary = {
            **self._observation_summary(obs),
            "camera_intrinsics_keys": sorted(intrinsics.keys()),
            "depth_keys": sorted(depth_maps.keys()),
            "qpos_arm": arm.tolist(),
            "qpos_gripper": gripper.tolist(),
            "task_text": task_text,
            "live_planning_enabled": self.live_planning_enabled,
            "require_plan": self.require_plan,
            "backend": self.backend,
        }
        (self._episode_dir / "observation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if not depth_maps:
            result = {
                "success": False,
                "joint_trajectory": None,
                "gripper_trajectory": None,
                "failure_reason": "no_depth_available",
                "debug": {
                    "failure_layer": "depth",
                    "rgb_camera_keys": sorted(images.keys()),
                    "intrinsics_keys": sorted(intrinsics.keys()),
                    "task_text_present": bool(task_text),
                    "qpos_shape": list(arm.shape),
                    "note": "MolmoSpaces policy observation did not expose depth; privileged simulator state was not used.",
                },
            }
            self._write_planning_result(result)
            return result

        if not intrinsics:
            result = {
                "success": False,
                "joint_trajectory": None,
                "gripper_trajectory": None,
                "failure_reason": "no_camera_intrinsics_available",
                "debug": {
                    "failure_layer": "observation",
                    "depth_keys": sorted(depth_maps.keys()),
                    "note": "Depth was present but no camera intrinsics were exposed for xyz reconstruction.",
                },
            }
            self._write_planning_result(result)
            return result

        result = {
            "success": False,
            "joint_trajectory": None,
            "gripper_trajectory": None,
            "failure_reason": "live_tiptop_pipeline_requires_m2t2_cutamp_integration",
            "debug": {
                "failure_layer": "cuTAMP",
                "depth_keys": sorted(depth_maps.keys()),
                "intrinsics_keys": sorted(intrinsics.keys()),
                "note": (
                    "Observation has depth/intrinsics, but the MolmoSpaces adapter still needs a "
                    "non-privileged TiPToP scene construction path for M2T2 and cuTAMP."
                ),
            },
        }
        self._write_planning_result(result)
        return result

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

    def _plan_result_to_queue(self, plan_result: dict[str, Any], observation: Any) -> dict[str, Any] | None:
        if not plan_result.get("success", False):
            return None
        joint_trajectory = np.asarray(plan_result.get("joint_trajectory"), dtype=np.float32)
        if joint_trajectory.ndim != 2 or joint_trajectory.shape[1] < int(self.action_spec.get("arm", 7)):
            raise ValueError(
                "plan_result.joint_trajectory must have shape [T, 7+] for MolmoSpaces joint-position actions"
            )
        gripper_trajectory = plan_result.get("gripper_trajectory")
        if gripper_trajectory is None:
            _, current_gripper = self._extract_qpos(observation)
            gripper_array = np.repeat(current_gripper.reshape(1, 1), joint_trajectory.shape[0], axis=0)
        else:
            gripper_array = np.asarray(gripper_trajectory, dtype=np.float32).reshape(-1, 1)
            if len(gripper_array) == 1 and len(joint_trajectory) > 1:
                gripper_array = np.repeat(gripper_array, len(joint_trajectory), axis=0)
            if len(gripper_array) != len(joint_trajectory):
                raise ValueError("gripper_trajectory length must match joint_trajectory length")

        for arm, gripper in zip(joint_trajectory, gripper_array):
            self.action_queue.push_action(arm, gripper)
        self._write_planning_result(plan_result)
        np.save(self._episode_dir / "trajectory.npy", joint_trajectory[:, : int(self.action_spec.get("arm", 7))])
        return {"steps": []}

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

    def _extract_depth_maps(self, observation: Any) -> dict[str, np.ndarray]:
        obs = self._first_observation(observation)
        if not isinstance(obs, dict):
            return {}
        candidates: dict[str, Any] = {}
        for key in ("depths", "depth_images"):
            value = obs.get(key)
            if isinstance(value, dict):
                candidates.update(value)
        for key in ("depth", "depth_image"):
            if key in obs:
                candidates[key] = obs[key]
        for key, value in obs.items():
            if isinstance(key, str) and "depth" in key.lower() and hasattr(value, "shape"):
                candidates[key] = value
        depth_maps: dict[str, np.ndarray] = {}
        for key, value in candidates.items():
            array = np.asarray(value)
            if array.ndim >= 2:
                depth_maps[str(key)] = array
        return depth_maps

    def _extract_camera_intrinsics(self, observation: Any) -> dict[str, np.ndarray]:
        obs = self._first_observation(observation)
        if not isinstance(obs, dict):
            return {}
        intrinsics: dict[str, np.ndarray] = {}
        for key, value in obs.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            if not key.startswith("sensor_param_"):
                continue
            camera_name = key.removeprefix("sensor_param_")
            matrix = value.get("intrinsic_cv", value.get("intrinsics", value.get("intrinsic_matrix")))
            if matrix is not None:
                intrinsics[camera_name] = np.asarray(matrix, dtype=np.float32)
        return intrinsics

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
            depth_maps = self._extract_depth_maps(obs)
            intrinsics = self._extract_camera_intrinsics(obs)
            has_depth = bool(depth_maps)
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
            "depth_keys": sorted(self._extract_depth_maps(obs).keys()),
            "camera_intrinsics_keys": sorted(self._extract_camera_intrinsics(obs).keys()),
            "robot_state_keys": robot_state_keys,
            "qpos_shape": list(arm.shape),
            "task_or_language": self._task_text(obs),
        }

    def _record_plan_result_metrics(self, plan_result: dict[str, Any], observation: Any) -> None:
        self.live_log_dir.mkdir(parents=True, exist_ok=True)
        success = bool(plan_result.get("success", False))
        fallback_reason = None if success else str(plan_result.get("failure_reason", "planning_failed"))
        metric = {
            "episode_id": self._episode_id,
            "planning_attempted": True,
            "planning_success": success,
            "fallback_used": not success,
            "fallback_reason": fallback_reason,
            "action_queue_length": int(len(self.action_queue)),
            "first_non_hold_action_step": self._first_non_hold_action_step(plan_result, observation) if success else None,
            "live_planning_enabled": self.live_planning_enabled,
            "require_plan": self.require_plan,
            "backend": self.backend,
            "m2t2_enabled": bool(plan_result.get("debug", {}).get("m2t2_enabled", False)),
            "cutamp_enabled": bool(plan_result.get("debug", {}).get("cutamp_enabled", False)),
        }
        with (self.live_log_dir / "episode_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(metric, sort_keys=True) + "\n")

    def _first_non_hold_action_step(self, plan_result: dict[str, Any], observation: Any) -> int | None:
        if not plan_result.get("success", False):
            return None
        arm, gripper = self._extract_qpos(observation)
        trajectory = np.asarray(plan_result.get("joint_trajectory"), dtype=np.float32)
        gripper_trajectory = plan_result.get("gripper_trajectory")
        gripper_array = None if gripper_trajectory is None else np.asarray(gripper_trajectory, dtype=np.float32).reshape(-1, 1)
        for idx, step in enumerate(trajectory):
            arm_changed = not np.allclose(step[: len(arm)], arm, atol=1e-6)
            gripper_changed = False
            if gripper_array is not None and idx < len(gripper_array):
                gripper_changed = not np.allclose(gripper_array[idx], gripper, atol=1e-6)
            if arm_changed or gripper_changed:
                return idx
        return None

    def _write_planning_failure(self, plan_result: dict[str, Any]) -> None:
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        payload = self._jsonable_plan_result(plan_result)
        (self._episode_dir / "planning_failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_planning_result(self, plan_result: dict[str, Any]) -> None:
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        payload = self._jsonable_plan_result(plan_result)
        (self._episode_dir / "planning_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _jsonable_plan_result(self, plan_result: dict[str, Any]) -> dict[str, Any]:
        payload = dict(plan_result)
        for key in ("joint_trajectory", "gripper_trajectory"):
            value = payload.get(key)
            if value is None:
                continue
            array = np.asarray(value)
            payload[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        return payload

    def _save_rgb_debug(self, name: str, image: Any) -> None:
        array = np.asarray(image)
        if array.ndim != 3:
            return
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        Image.fromarray(array).save(self._episode_dir / f"rgb_{self._safe_name(name)}.png")

    def _safe_name(self, name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name))

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
