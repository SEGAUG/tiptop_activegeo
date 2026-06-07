from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

_log = logging.getLogger(__name__)

_SAMPLE_DIR_RE = re.compile(r"sample_(\d+)")


def load_qwen_aligned_detection(path: str | Path) -> dict[str, Any]:
    """Load Qwen output after conversion to TiPToP/Gemini-aligned format."""
    detection = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(detection.get("bboxes"), list):
        raise ValueError(f"{path} must contain a list field named 'bboxes'")
    if not isinstance(detection.get("grounded_atoms"), list):
        raise ValueError(f"{path} must contain a list field named 'grounded_atoms'")
    return detection


def discover_manual_multiview_h5_paths(run_dir: str | Path) -> list[Path]:
    """Find calibrated H5 observations saved by piper-manual-multiview-capture."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Manual multiview run not found: {run_dir}")
    paths = sorted(
        run_dir.glob("sample_*/observation_calibrated.h5"),
        key=lambda path: _sample_sort_key(path),
    )
    if not paths:
        raise ValueError(f"No sample_*/observation_calibrated.h5 files found under {run_dir}")
    return paths


def parse_multiview_h5_paths(value: str | None) -> list[Path]:
    """Parse comma-separated H5 paths from the CLI."""
    if value is None or not value.strip():
        return []
    return [Path(token.strip()) for token in value.split(",") if token.strip()]


def resolve_multiview_h5_paths(
    manual_multiview_run: str | None = None,
    multiview_h5_paths: str | None = None,
) -> list[Path]:
    """Resolve all extra H5 observations to append before M2T2 point cloud construction."""
    resolved: list[Path] = []
    if manual_multiview_run:
        resolved.extend(discover_manual_multiview_h5_paths(manual_multiview_run))
    resolved.extend(parse_multiview_h5_paths(multiview_h5_paths))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        normalized = path.expanduser()
        if normalized in seen:
            continue
        if not normalized.exists():
            raise FileNotFoundError(f"Multiview H5 not found: {normalized}")
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _sample_sort_key(path: Path) -> tuple[int, str]:
    match = _SAMPLE_DIR_RE.fullmatch(path.parent.name)
    if match:
        return int(match.group(1)), path.parent.name
    return 10**9, path.parent.name


def _depth_to_xyz(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def _normalize_gripper_mask(gripper_mask: np.ndarray | None, image_shape: tuple[int, int]) -> np.ndarray | None:
    if gripper_mask is None:
        return None
    mask = np.asarray(gripper_mask).astype(bool)
    if mask.ndim == 3:
        mask = mask.any(axis=-1)
    if mask.shape != image_shape:
        raise ValueError(f"gripper_mask shape {mask.shape} does not match depth shape {image_shape}")
    return mask


def rgbd_to_world_maps(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_from_cam: np.ndarray,
    gripper_mask: np.ndarray | None = None,
    depth_trunc_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project an RGB-D frame into a structured world-frame XYZ/RGB map."""
    if depth_trunc_m is None:
        from tiptop.config import tiptop_cfg

        depth_trunc_m = float(tiptop_cfg().perception.depth_trunc_m)
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0
    depth[depth > float(depth_trunc_m)] = 0.0
    xyz_map = _depth_to_xyz(depth, intrinsics.astype(np.float32))
    world_from_cam = np.asarray(world_from_cam, dtype=np.float32)
    xyz_map = xyz_map @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]
    mask = _normalize_gripper_mask(gripper_mask, depth.shape)
    if mask is not None:
        xyz_map[mask] = 0.0
    rgb_map = rgb.astype(np.float32) / 255.0
    return xyz_map.astype(np.float32), rgb_map.astype(np.float32)


def load_multiview_h5_rgbd(path: str | Path) -> dict[str, np.ndarray]:
    """Load the RGB-D arrays and calibrated camera pose from a saved observation H5."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        rgb = f["rgb"][:]
        depth = f["depth"][:]
        intrinsics = f["intrinsic_matrix"][:]
        if "world_from_cam" in f:
            world_from_cam = f["world_from_cam"][:]
        else:
            pos_w = f["pos_w"][:]
            quat_w_ros = f["quat_w_ros"][:]
            quat_xyzw = np.array([quat_w_ros[1], quat_w_ros[2], quat_w_ros[3], quat_w_ros[0]], dtype=np.float32)
            world_from_cam = np.eye(4, dtype=np.float32)
            world_from_cam[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
            world_from_cam[:3, 3] = pos_w.astype(np.float32)
        calibrated = bool(f.attrs.get("world_from_cam_calibrated", False))
    if not calibrated:
        raise ValueError(f"Refusing to use uncalibrated multiview H5 for fused point cloud: {path}")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    return {
        "rgb": rgb.astype(np.uint8),
        "depth": depth.astype(np.float32),
        "intrinsic_matrix": intrinsics.astype(np.float32),
        "world_from_cam": world_from_cam.astype(np.float32),
    }


def _flatten_valid_points(xyz_map: np.ndarray, rgb_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz_flat = xyz_map.reshape(-1, 3)
    rgb_flat = rgb_map.reshape(-1, 3)
    valid = np.isfinite(xyz_flat).all(axis=1)
    valid &= np.linalg.norm(xyz_flat, axis=1) > 1e-6
    return xyz_flat[valid], rgb_flat[valid]


async def predict_multiview_grasps(
    session,
    frame,
    current_world_from_cam: np.ndarray,
    multiview_h5_paths: list[Path],
    downsample_voxel_size: float,
    gripper_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fuse saved RGB-D views plus the current live view before calling M2T2."""
    from tiptop.config import tiptop_cfg
    from tiptop.perception.m2t2 import generate_grasps_async
    from tiptop.perception.utils import get_o3d_pcd

    if frame.depth is None:
        raise RuntimeError("Current frame has no depth; cannot build multiview M2T2 point cloud")

    source_xyz_maps = []
    source_rgb_maps = []
    source_names = ["current_live_view"]
    source_point_counts = []

    current_xyz_map, current_rgb_map = rgbd_to_world_maps(
        frame.rgb,
        frame.depth,
        frame.intrinsics,
        current_world_from_cam,
        gripper_mask=gripper_mask,
    )
    current_xyz_flat, current_rgb_flat = _flatten_valid_points(current_xyz_map, current_rgb_map)
    source_xyz_maps.append(current_xyz_flat)
    source_rgb_maps.append(current_rgb_flat)
    source_point_counts.append(int(len(current_xyz_flat)))

    for h5_path in multiview_h5_paths:
        observation = load_multiview_h5_rgbd(h5_path)
        xyz_map, rgb_map = rgbd_to_world_maps(
            observation["rgb"],
            observation["depth"],
            observation["intrinsic_matrix"],
            observation["world_from_cam"],
            gripper_mask=gripper_mask,
        )
        xyz_flat, rgb_flat = _flatten_valid_points(xyz_map, rgb_map)
        source_xyz_maps.append(xyz_flat)
        source_rgb_maps.append(rgb_flat)
        source_names.append(str(h5_path))
        source_point_counts.append(int(len(xyz_flat)))

    scene_xyz = np.concatenate(source_xyz_maps, axis=0)
    scene_rgb = np.concatenate(source_rgb_maps, axis=0)
    if len(scene_xyz) == 0:
        raise RuntimeError("No valid RGB-D points found across current and saved multiview observations")

    pcd = await asyncio.to_thread(
        get_o3d_pcd,
        scene_xyz,
        scene_rgb,
        downsample_voxel_size,
    )
    xyz_downsampled = np.asarray(pcd.points)
    rgb_downsampled = np.asarray(pcd.colors)

    cfg = tiptop_cfg()
    grasps = await generate_grasps_async(
        session,
        cfg.perception.m2t2.url,
        scene_xyz=xyz_downsampled,
        scene_rgb=rgb_downsampled,
        apply_bounds=cfg.perception.m2t2.apply_bounds,
        request_timeout_s=float(cfg.perception.m2t2.get("request_timeout_s", 30.0)),
    )

    return {
        "depth_map": frame.depth,
        "xyz_map": current_xyz_map,
        "rgb_map": current_rgb_map,
        "xyz_downsampled": xyz_downsampled,
        "rgb_downsampled": rgb_downsampled,
        "pcd_downsampled": pcd,
        "grasps": grasps,
        "multiview_source_names": source_names,
        "multiview_source_point_counts": source_point_counts,
        "multiview_num_points_before_downsample": int(len(scene_xyz)),
        "multiview_num_points_after_downsample": int(len(xyz_downsampled)),
    }


def _empty_grasp_group() -> dict[str, np.ndarray]:
    return {
        "poses": np.array([]).reshape(0, 4, 4),
        "confidences": np.array([]),
        "contacts": np.array([]).reshape(0, 3),
    }


def filter_grasps_by_detection_masks(
    xyz_map: np.ndarray,
    masks: np.ndarray,
    bboxes: list[dict[str, Any]],
    grasps: dict[str, dict[str, np.ndarray]],
    contact_threshold: float | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Associate raw M2T2 grasp candidates to SAM-detected objects by contact proximity."""
    if contact_threshold is None:
        from tiptop.config import tiptop_cfg

        threshold = float(tiptop_cfg().perception.contact_threshold_m)
    else:
        threshold = float(contact_threshold)
    masks_2d = masks.squeeze(1).astype(bool)
    object_points: dict[str, np.ndarray] = {}

    for mask_2d, bbox in zip(masks_2d, bboxes):
        label = bbox["label"]
        points = xyz_map[mask_2d]
        valid = np.isfinite(points).all(axis=1)
        valid &= np.linalg.norm(points, axis=1) > 1e-6
        object_points[label] = points[valid]

    filtered = {bbox["label"]: _empty_grasp_group() for bbox in bboxes}
    nonempty_labels = [label for label, points in object_points.items() if len(points) > 0]
    if not nonempty_labels:
        return filtered

    all_object_points = np.vstack([object_points[label] for label in nonempty_labels])
    point_to_label = np.concatenate([[label] * len(object_points[label]) for label in nonempty_labels])
    object_kdtree = KDTree(all_object_points)

    grouped: dict[str, dict[str, list[np.ndarray]]] = {
        label: {"poses": [], "confidences": [], "contacts": []} for label in filtered
    }
    for grasp_group in grasps.values():
        poses = np.asarray(grasp_group.get("poses", []))
        confidences = np.asarray(grasp_group.get("confidences", []))
        contacts = np.asarray(grasp_group.get("contacts", []))
        if len(poses) == 0 or len(contacts) == 0:
            continue

        contact_points = contacts.reshape(len(contacts), -1, 3).mean(axis=1)
        distances, nearest_indices = object_kdtree.query(contact_points)
        for idx, (distance, nearest_index) in enumerate(zip(distances, nearest_indices)):
            if distance >= threshold:
                continue
            label = point_to_label[nearest_index]
            grouped[label]["poses"].append(poses[idx])
            grouped[label]["confidences"].append(confidences[idx])
            grouped[label]["contacts"].append(contact_points[idx])

    for label, values in grouped.items():
        if not values["poses"]:
            continue
        order = np.argsort(np.asarray(values["confidences"]))[::-1]
        filtered[label] = {
            "poses": np.asarray(values["poses"])[order],
            "confidences": np.asarray(values["confidences"])[order],
            "contacts": np.asarray(values["contacts"])[order],
        }
    return filtered


async def _run(
    qwen_aligned_json: str,
    bridge_url: str,
    output_dir: str,
    rr_spawn: bool,
    m2t2_apply_bounds: bool | None,
    m2t2_request_timeout_s: float | None,
    camera_serial: str | None,
    manual_multiview_run: str | None,
    multiview_h5_paths: str | None,
    world_frame: Literal["identity", "calibrated"],
) -> None:
    import aiohttp
    import cv2
    import torch
    from PIL import Image

    from tiptop.config import tiptop_cfg
    from tiptop.perception.cameras import Frame
    from tiptop.perception.sam2 import sam2_segment_objects
    from tiptop.perception.utils import get_o3d_pcd
    from tiptop.perception.visualization import visualize_detections, visualize_masks
    from tiptop.perception_wrapper import predict_depth_and_grasps
    from tiptop.piper import PiperRobotClient
    from tiptop.scripts.piper_capture_snapshot import _compute_world_from_cam
    from tiptop.utils import add_file_handler, load_gripper_mask_for_image_shape, remove_file_handler

    del rr_spawn  # kept for CLI symmetry with piper-perception-dry-run
    if m2t2_apply_bounds is not None:
        tiptop_cfg().perception.m2t2.apply_bounds = bool(m2t2_apply_bounds)
    if m2t2_request_timeout_s is not None:
        tiptop_cfg().perception.m2t2.request_timeout_s = float(m2t2_request_timeout_s)
    multiview_paths = resolve_multiview_h5_paths(
        manual_multiview_run=manual_multiview_run,
        multiview_h5_paths=multiview_h5_paths,
    )
    multiview_enabled = len(multiview_paths) > 0
    camera_serial = camera_serial or str(tiptop_cfg().cameras.hand.serial)

    detection = load_qwen_aligned_detection(qwen_aligned_json)
    bboxes = detection["bboxes"]
    grounded_atoms = detection["grounded_atoms"]

    client = PiperRobotClient(base_url=bridge_url, timeout_s=10.0)
    snapshot = client.get_snapshot()
    q_init = client.get_joint_positions()

    timestamp = datetime.now()
    save_dir = Path(output_dir) / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    perception_dir = save_dir / "perception"
    perception_dir.mkdir(exist_ok=True)
    file_handler = add_file_handler(save_dir / "piper_qwen_m2t2_dry_run.log")

    try:
        rgb = snapshot["rgb"].astype(np.uint8)
        depth = snapshot["depth"].astype(np.float32)
        intrinsics = snapshot["intrinsic_matrix"].astype(np.float32)
        frame = Frame(
            serial="piper_runtime_bridge",
            timestamp=time.time(),
            rgb=rgb,
            intrinsics=intrinsics,
            depth=depth,
        )
        gripper_mask = load_gripper_mask_for_image_shape(rgb.shape)
        rgb_pil = Image.fromarray(rgb)

        _log.info("Running SAM2 with %d Qwen/Gemini-aligned boxes", len(bboxes))
        masks = await asyncio.to_thread(sam2_segment_objects, rgb_pil, bboxes)

        world_frame_calibrated = multiview_enabled or world_frame == "calibrated"
        if world_frame_calibrated:
            world_from_cam, calibration_metadata = _compute_world_from_cam(q_init, camera_serial)
        else:
            # Preserve legacy single-view dry-run behavior unless multiview calibrated H5s are requested.
            world_from_cam = np.eye(4, dtype=np.float32)
            calibration_metadata = None
        connector = aiohttp.TCPConnector(limit=10, force_close=True)
        timeout = aiohttp.ClientTimeout(total=180.0)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            if multiview_enabled:
                _log.info(
                    "Running M2T2 grasp prediction from %d saved views plus current live view",
                    len(multiview_paths),
                )
                depth_results = await predict_multiview_grasps(
                    session,
                    frame,
                    world_from_cam,
                    multiview_paths,
                    tiptop_cfg().perception.voxel_downsample_size,
                    gripper_mask=gripper_mask,
                )
            else:
                _log.info("Running M2T2 grasp prediction from Piper bridge depth")
                depth_results = await predict_depth_and_grasps(
                    session,
                    frame,
                    world_from_cam,
                    tiptop_cfg().perception.voxel_downsample_size,
                    depth_estimator=None,
                    gripper_mask=gripper_mask,
                )

        cv2.imwrite(str(save_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(perception_dir / "depth.png"), depth_mm)

        bbox_viz = visualize_detections(rgb_pil, bboxes, output_path=str(save_dir / "bboxes_viz.png"), show_plot=False)
        masks_viz = visualize_masks(rgb_pil, masks, bboxes)
        cv2.imwrite(str(save_dir / "masks_viz.png"), cv2.cvtColor(masks_viz, cv2.COLOR_RGB2BGR))

        (perception_dir / "bboxes.json").write_text(json.dumps(bboxes, indent=2), encoding="utf-8")
        (perception_dir / "grounded_atoms.json").write_text(json.dumps(grounded_atoms, indent=2), encoding="utf-8")
        np.savez_compressed(perception_dir / "masks.npz", masks > 0.5)
        if gripper_mask is not None:
            Image.fromarray(gripper_mask.astype(np.uint8) * 255).save(perception_dir / "gripper_mask.png")
        np.savez_compressed(
            perception_dir / "snapshot_arrays.npz",
            rgb=rgb,
            depth=depth,
            intrinsic_matrix=intrinsics,
            q_init=q_init,
            world_from_cam=world_from_cam,
            xyz_map=depth_results["xyz_map"],
            rgb_map=depth_results["rgb_map"],
            xyz_downsampled=depth_results["xyz_downsampled"],
            rgb_downsampled=depth_results["rgb_downsampled"],
        )
        pcd = get_o3d_pcd(depth_results["xyz_downsampled"], depth_results["rgb_downsampled"])
        import open3d as o3d

        o3d.io.write_point_cloud(str(perception_dir / "pointcloud_downsampled.ply"), pcd)
        filtered_grasps = filter_grasps_by_detection_masks(
            depth_results["xyz_map"],
            masks > 0.5,
            bboxes,
            depth_results["grasps"],
        )
        torch.save(depth_results["grasps"], perception_dir / "grasps_raw.pt")
        torch.save(filtered_grasps, perception_dir / "grasps.pt")

        grasp_counts = {
            label: int(len(group.get("poses", [])))
            for label, group in filtered_grasps.items()
        }
        raw_grasp_counts = {
            label: int(len(group.get("poses", [])))
            for label, group in depth_results["grasps"].items()
        }
        metadata = {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "bridge_url": bridge_url,
            "qwen_aligned_json": str(qwen_aligned_json),
            "q_init": q_init.tolist(),
            "bboxes": bboxes,
            "grounded_atoms": grounded_atoms,
            "num_grasp_groups": len(filtered_grasps),
            "grasp_counts": grasp_counts,
            "num_raw_grasp_groups": len(depth_results["grasps"]),
            "raw_grasp_counts": raw_grasp_counts,
            "grasp_filter_contact_threshold_m": float(tiptop_cfg().perception.contact_threshold_m),
            "gripper_mask_applied": gripper_mask is not None,
            "camera_serial": camera_serial,
            "m2t2_multiview_enabled": bool(multiview_enabled),
            "m2t2_world_frame_mode": "calibrated" if world_frame_calibrated else "identity",
            "m2t2_current_world_from_cam": world_from_cam.tolist(),
            "m2t2_world_from_cam_calibrated": bool(world_frame_calibrated),
            "m2t2_multiview_h5_paths": [str(path) for path in multiview_paths],
            "m2t2_multiview_source_names": depth_results.get("multiview_source_names", ["current_live_view"]),
            "m2t2_multiview_source_point_counts": depth_results.get("multiview_source_point_counts", []),
            "m2t2_num_points_before_downsample": int(
                depth_results.get("multiview_num_points_before_downsample", len(depth_results["xyz_downsampled"]))
            ),
            "m2t2_num_points_after_downsample": int(len(depth_results["xyz_downsampled"])),
            "calibration": calibration_metadata,
            "safe_for_robot_execution": False,
            "reason_not_safe": "Qwen/M2T2 dry-run does not execute robot motion",
        }
        (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"save_dir": str(save_dir), **metadata}, indent=2))
        if bbox_viz is None:
            _log.warning("No bbox visualization was generated")
    finally:
        remove_file_handler(file_handler)


def piper_qwen_m2t2_dry_run(
    qwen_aligned_json: str,
    bridge_url: str = "http://127.0.0.1:8766",
    output_dir: str = "/data/data2/jinhui.lin/code/aicode/piper_real_outputs/qwen_m2t2_dry_run",
    rr_spawn: bool = False,
    m2t2_apply_bounds: bool | None = None,
    m2t2_request_timeout_s: float | None = None,
    camera_serial: str | None = None,
    manual_multiview_run: str | None = None,
    multiview_h5_paths: str | None = None,
    world_frame: Literal["identity", "calibrated"] = "identity",
) -> None:
    """Run SAM2 and M2T2 using a saved Qwen Gemini-aligned detection JSON."""
    from tiptop.utils import setup_logging

    setup_logging(level=logging.INFO)
    asyncio.run(
        _run(
            qwen_aligned_json,
            bridge_url,
            output_dir,
            rr_spawn,
            m2t2_apply_bounds,
            m2t2_request_timeout_s,
            camera_serial,
            manual_multiview_run,
            multiview_h5_paths,
            world_frame,
        )
    )


def entrypoint() -> None:
    import tyro

    tyro.cli(piper_qwen_m2t2_dry_run)


if __name__ == "__main__":
    entrypoint()
