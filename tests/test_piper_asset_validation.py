from tiptop.piper.asset_validation import validate_asset_debug_payload


def test_validate_asset_debug_payload_accepts_clear_asset():
    payload = {
        "link_sphere_count": 42,
        "gripper_sphere_count": 6,
        "near_pairs": [{"margin_m": 0.012}, {"margin_m": 0.02}],
    }
    report = validate_asset_debug_payload(payload, min_link_spheres=20, min_gripper_spheres=4)
    assert report["asset_ready_for_planning_dry_run"] is True
    assert report["min_self_collision_margin_m"] == 0.012


def test_validate_asset_debug_payload_rejects_missing_gripper_spheres():
    payload = {"link_sphere_count": 42, "gripper_sphere_count": 0, "near_pairs": []}
    report = validate_asset_debug_payload(payload, min_link_spheres=20, min_gripper_spheres=4)
    assert report["asset_ready_for_planning_dry_run"] is False
    assert "gripper_sphere_count" in report["reasons"][0]


def test_validate_asset_debug_payload_rejects_intersecting_spheres():
    payload = {"link_sphere_count": 42, "gripper_sphere_count": 6, "near_pairs": [{"margin_m": -0.001}]}
    report = validate_asset_debug_payload(payload, min_link_spheres=20, min_gripper_spheres=4)
    assert report["asset_ready_for_planning_dry_run"] is False
    assert "self-collision" in " ".join(report["reasons"])
