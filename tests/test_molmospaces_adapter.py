import json
from pathlib import Path

import numpy as np


def _fake_observation():
    return {
        "images": {
            "exo_camera_1": np.zeros((4, 5, 3), dtype=np.uint8),
            "wrist_camera": np.zeros((2, 3, 3), dtype=np.uint8),
        },
        "depths": {"exo_camera_1": np.zeros((4, 5), dtype=np.float32)},
        "robot_state": {"qpos": np.arange(8, dtype=np.float64)},
        "task": "pick up the mug",
    }


def test_action_queue_splits_tiptop_serialized_trajectory():
    from tiptop_molmospaces.action_queue import ActionQueue

    queue = ActionQueue(action_spec={"arm": 7, "gripper": 1}, default_gripper=255.0)
    queue.extend_plan(
        {
            "steps": [
                {"type": "trajectory", "positions": [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]]},
                {"type": "gripper", "action": "close"},
            ]
        }
    )

    assert len(queue) == 3
    first = queue.pop()
    assert first["arm"].shape == (7,)
    assert first["gripper"].shape == (1,)
    np.testing.assert_allclose(first["arm"], np.arange(7, dtype=np.float32))
    np.testing.assert_allclose(first["gripper"], np.array([255.0], dtype=np.float32))
    np.testing.assert_allclose(queue.pop()["arm"], np.arange(7, 14, dtype=np.float32))
    np.testing.assert_allclose(queue.pop()["gripper"], np.array([0.0], dtype=np.float32))


def test_policy_falls_back_to_hold_action_and_writes_schema(tmp_path):
    from tiptop_molmospaces.configs import TiPToPEvalConfig
    from tiptop_molmospaces.policy import TiPToPPolicy

    config = TiPToPEvalConfig()
    config.policy_config.log_dir = str(tmp_path)
    policy = TiPToPPolicy(config, task=None)
    policy.reset()

    action = policy.get_action(_fake_observation())

    assert set(action) == {"arm", "gripper"}
    assert action["arm"].shape == (7,)
    assert action["gripper"].shape == (1,)
    np.testing.assert_allclose(action["arm"], np.arange(7, dtype=np.float32))
    np.testing.assert_allclose(action["gripper"], np.array([7.0], dtype=np.float32))

    schema_path = Path(tmp_path) / "observation_schema.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text())
    assert schema["type"] == "dict"
    assert "robot_state" in schema["keys"]
    assert schema["keys"]["images"]["keys"]["exo_camera_1"]["shape"] == [4, 5, 3]


def test_eval_config_points_to_policy_and_joint_position_action_type():
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from tiptop_molmospaces.configs import TiPToPEvalConfig
    from tiptop_molmospaces.policy import TiPToPPolicy

    config = TiPToPEvalConfig()

    assert config.policy_config.policy_cls is TiPToPPolicy
    assert config.policy_config.camera_names == ["exo_camera_1", "wrist_camera"]
    assert config.policy_config.action_move_group_names == ["arm", "gripper"]
    assert config.policy_config.action_spec == {"arm": 7, "gripper": 1}
    assert config.policy_config.action_type == "joint_position"
    assert isinstance(config.robot_config, FrankaRobotConfig)
    assert config.policy_dt_ms == 200.0
