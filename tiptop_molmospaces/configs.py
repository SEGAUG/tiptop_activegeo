from __future__ import annotations

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


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
