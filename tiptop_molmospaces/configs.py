from __future__ import annotations

import logging
import os
from typing import Any

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig

log = logging.getLogger(__name__)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _install_recorded_camera_depth_patch(camera_names: list[str]) -> None:
    """Enable MolmoSpaces' built-in camera DepthSensor for replayed benchmark cameras.

    JSON benchmarks replay their recorded camera specs and default to
    ``record_depth=False``. This wrapper flips ``record_depth`` only when the
    external adapter explicitly opts in via ``TIPTOP_ENABLE_RENDERED_DEPTH=1``.
    It still uses the current policy cameras and MolmoSpaces' camera renderer;
    it does not read benchmark object poses or target poses.
    """

    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

    if getattr(JsonEvalTaskSampler, "_tiptop_rendered_depth_patch", False):
        JsonEvalTaskSampler._tiptop_depth_camera_names = set(camera_names)
        return

    original = JsonEvalTaskSampler._build_camera_config_from_spec

    def _build_camera_config_with_tiptop_depth(self: Any, episode_spec: Any):
        camera_config = original(self, episode_spec)
        enabled = _env_enabled("TIPTOP_ENABLE_RENDERED_DEPTH", False)
        if enabled:
            target_names = getattr(JsonEvalTaskSampler, "_tiptop_depth_camera_names", set(camera_names))
            for camera in camera_config.cameras:
                if not target_names or camera.name in target_names:
                    camera.record_depth = True
            log.info(
                "Enabled rendered MolmoSpaces camera depth for TiPToP cameras: %s",
                sorted(target_names) if target_names else "all",
            )
        return camera_config

    JsonEvalTaskSampler._build_camera_config_from_spec = _build_camera_config_with_tiptop_depth
    JsonEvalTaskSampler._tiptop_rendered_depth_patch = True
    JsonEvalTaskSampler._tiptop_depth_camera_names = set(camera_names)


class TiPToPPolicyConfig(BasePolicyConfig):
    policy_cls: type | None = None
    policy_type: str = "learned"

    checkpoint_path: str | None = None
    camera_names: list[str] = ["exo_camera_1", "wrist_camera"]
    action_move_group_names: list[str] = ["arm", "gripper"]
    action_spec: dict[str, int] = {"arm": 7, "gripper": 1}
    action_type: str = "joint_position"
    log_dir: str = "logs"
    serialized_plan_path: str | None = None
    enable_tiptop_planning: bool = False
    enable_rendered_depth: bool = False
    depth_backend: str = "rendered"
    safe_gripper_action: float = 255.0

    def model_post_init(self, __context) -> None:
        from tiptop_molmospaces.policy import TiPToPPolicy

        self.policy_cls = TiPToPPolicy


class TiPToPEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: TiPToPPolicyConfig = TiPToPPolicyConfig()
    policy_dt_ms: float = 200.0

    @property
    def tag(self) -> str:
        return "tiptop_activegeometry_molmospaces"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)
        if self.policy_config.enable_rendered_depth or _env_enabled("TIPTOP_ENABLE_RENDERED_DEPTH", False):
            _install_recorded_camera_depth_patch(self.policy_config.camera_names)


class TiPToPDepthEvalConfig(TiPToPEvalConfig):
    policy_config: TiPToPPolicyConfig = TiPToPPolicyConfig(enable_rendered_depth=True)

    @property
    def tag(self) -> str:
        return "tiptop_activegeometry_molmospaces_depth"
