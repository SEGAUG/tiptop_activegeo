from tiptop.piper.wrist_calibration_readiness import summarize_charuco_readiness


def test_summarize_charuco_readiness_accepts_good_sample():
    payload = {
        "samples": [
            {"detected": True, "charuco_corners": 70, "aruco_markers": 44, "camera_serial": "243722072079"}
        ]
    }
    summary = summarize_charuco_readiness(payload, min_corners=50, min_markers=20)
    assert summary["ready"] is True
    assert summary["best_charuco_corners"] == 70
    assert summary["sample_count"] == 1


def test_summarize_charuco_readiness_rejects_missing_board():
    payload = {"samples": [{"detected": False, "charuco_corners": 0, "aruco_markers": 0}]}
    summary = summarize_charuco_readiness(payload, min_corners=50, min_markers=20)
    assert summary["ready"] is False
    assert "not enough ChArUco corners" in summary["reason"]


def test_summarize_charuco_readiness_counts_detected_samples():
    payload = {
        "samples": [
            {"detected": False, "charuco_corners": 0, "aruco_markers": 0},
            {"detected": True, "charuco_corners": 55, "aruco_markers": 22},
        ]
    }
    summary = summarize_charuco_readiness(payload, min_corners=50, min_markers=20)
    assert summary["detected_sample_count"] == 1
    assert summary["ready"] is True
