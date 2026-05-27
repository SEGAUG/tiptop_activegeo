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
