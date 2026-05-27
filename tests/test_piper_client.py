import numpy as np
import pytest

from tiptop.piper.piper_client import PiperClientError, PiperRobotClient


class StaticPiperClient(PiperRobotClient):
    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)


def test_validate_joint_path_rejects_wrong_shape():
    client = StaticPiperClient()
    with pytest.raises(PiperClientError, match="Expected joint path shape"):
        client._validate_joint_path(np.zeros((3, 7)), np.ones(3))


def test_validate_joint_path_rejects_large_initial_error():
    client = StaticPiperClient(max_initial_error_rad=0.01)
    path = np.array([[0.02, 0, 0, 0, 0, 0]], dtype=np.float32)
    with pytest.raises(PiperClientError, match="First waypoint"):
        client._validate_joint_path(path, np.ones(1))


def test_execute_joint_impedance_path_clamps_duration(monkeypatch):
    captured = {}

    class CapturingClient(StaticPiperClient):
        def _request_json(self, method, path, payload=None, timeout=None):
            captured.update({"method": method, "path": path, "payload": payload, "timeout": timeout})
            return {"success": True}

    client = CapturingClient(min_waypoint_dt_s=0.5)
    result = client.execute_joint_impedance_path(np.zeros((2, 6)), durations=[0.01, 0.2])

    assert result == {"success": True}
    assert captured["method"] == "POST"
    assert captured["path"] == "/execute_joint_path"
    assert captured["payload"]["durations"] == [0.5, 0.5]


def test_set_enabled_posts_enable_payload():
    captured = {}

    class CapturingClient(StaticPiperClient):
        def _request_json(self, method, path, payload=None, timeout=None):
            captured.update({"method": method, "path": path, "payload": payload})
            return {"success": True}

    result = CapturingClient().set_enabled(True)

    assert result == {"success": True}
    assert captured == {"method": "POST", "path": "/enable", "payload": {"enable": True}}


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
