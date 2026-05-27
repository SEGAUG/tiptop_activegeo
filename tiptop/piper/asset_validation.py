from __future__ import annotations


def validate_asset_debug_payload(
    payload: dict,
    *,
    min_link_spheres: int = 20,
    min_gripper_spheres: int = 4,
    min_allowed_margin_m: float = 0.0,
) -> dict:
    reasons: list[str] = []
    link_count = int(payload.get("link_sphere_count", 0))
    gripper_count = int(payload.get("gripper_sphere_count", 0))
    near_pairs = list(payload.get("near_pairs", []))
    margins = [float(row.get("margin_m", 0.0)) for row in near_pairs]
    min_margin = min(margins) if margins else None

    if link_count < min_link_spheres:
        reasons.append(f"link_sphere_count {link_count} < {min_link_spheres}")
    if gripper_count < min_gripper_spheres:
        reasons.append(f"gripper_sphere_count {gripper_count} < {min_gripper_spheres}")
    if min_margin is not None and min_margin < min_allowed_margin_m:
        reasons.append(f"near self-collision margin {min_margin:.6f} m < {min_allowed_margin_m:.6f} m")

    return {
        "asset_ready_for_planning_dry_run": len(reasons) == 0,
        "reasons": reasons,
        "link_sphere_count": link_count,
        "gripper_sphere_count": gripper_count,
        "near_pair_count": len(near_pairs),
        "min_self_collision_margin_m": min_margin,
        "min_link_spheres": int(min_link_spheres),
        "min_gripper_spheres": int(min_gripper_spheres),
        "min_allowed_margin_m": float(min_allowed_margin_m),
    }
