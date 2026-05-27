from __future__ import annotations


def summarize_charuco_readiness(payload: dict, *, min_corners: int = 50, min_markers: int = 20) -> dict:
    samples = list(payload.get("samples", []))
    detected = [sample for sample in samples if sample.get("detected") is True]
    best_corners = max([int(sample.get("charuco_corners", 0)) for sample in samples], default=0)
    best_markers = max([int(sample.get("aruco_markers", 0)) for sample in samples], default=0)

    ready = best_corners >= min_corners and best_markers >= min_markers
    if ready:
        reason = "ChArUco detection is sufficient for calibration sampling"
    elif best_corners < min_corners:
        reason = f"not enough ChArUco corners: {best_corners} < {min_corners}"
    else:
        reason = f"not enough ArUco markers: {best_markers} < {min_markers}"

    return {
        "ready": ready,
        "reason": reason,
        "sample_count": len(samples),
        "detected_sample_count": len(detected),
        "best_charuco_corners": best_corners,
        "best_aruco_markers": best_markers,
        "min_corners": int(min_corners),
        "min_markers": int(min_markers),
    }
