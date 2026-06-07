from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def extract_task_text(observation_or_summary: Any) -> str:
    if not isinstance(observation_or_summary, dict):
        raise ValueError("task_text_missing: observation is not a dict")

    for key in ("task_text", "task_or_language", "task", "text", "instruction", "language", "task_description"):
        value = observation_or_summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested in ("task_description", "text", "instruction", "language"):
                nested_value = value.get(nested)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()

    task_info = observation_or_summary.get("task_info")
    if isinstance(task_info, dict):
        for key in ("task_description", "text", "instruction", "language"):
            value = task_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise ValueError("task_text_missing")


def select_primary_camera(observation: dict[str, Any]) -> str:
    override = os.environ.get("TIPTOP_PRIMARY_CAMERA")
    if override:
        if override not in observation:
            raise ValueError(f"primary_camera_missing: {override}")
        return override
    if "exo_camera_1" in observation:
        return "exo_camera_1"
    if "wrist_camera" in observation:
        return "wrist_camera"
    for key, value in observation.items():
        if isinstance(key, str) and "camera" in key and "depth" not in key.lower() and hasattr(value, "shape"):
            return key
    raise ValueError("primary_camera_missing")


def _as_uint8_rgb(rgb: Any) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected RGB image shape [H,W,3], got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def run_qwen_grounding(rgb: Any, task_text: str, camera_name: str, debug_dir: Path) -> dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    if not _env_enabled("TIPTOP_ENABLE_QWEN_GROUNDING", False):
        result = {"success": False, "reason": "qwen_disabled", "camera_name": camera_name, "detections": []}
        _write_json(debug_dir / "qwen_detections.json", result)
        return result
    if not os.environ.get("DASHSCOPE_API_KEY"):
        result = {"success": False, "reason": "missing_api_key", "camera_name": camera_name, "detections": []}
        _write_json(debug_dir / "qwen_raw_response.json", result)
        _write_json(debug_dir / "qwen_detections.json", result)
        return result

    try:
        from tiptop.perception.qwen_vl import qwen_detect_and_translate_raw

        image = Image.fromarray(_as_uint8_rgb(rgb))
        raw_text, parsed, aligned = qwen_detect_and_translate_raw(image, task_text)
        payload = {
            "success": True,
            "camera_name": camera_name,
            "task_text": task_text,
            "raw_text": raw_text,
            "parsed": parsed,
            "aligned": aligned,
        }
        _write_json(debug_dir / "qwen_raw_response.json", payload)
        _write_json(debug_dir / "qwen_detections.json", payload)
        return payload
    except Exception as exc:
        result = {
            "success": False,
            "reason": "qwen_request_failed",
            "camera_name": camera_name,
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "detections": [],
        }
        _write_json(debug_dir / "qwen_raw_response.json", result)
        _write_json(debug_dir / "qwen_detections.json", result)
        return result


def _coerce_bbox_entries(qwen_result: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if isinstance(qwen_result.get("detections"), list):
        return qwen_result["detections"], str(qwen_result.get("bbox_format", "xyxy"))
    if isinstance(qwen_result.get("parsed"), dict) and isinstance(qwen_result["parsed"].get("bboxes"), list):
        return qwen_result["parsed"]["bboxes"], "xyxy"
    if isinstance(qwen_result.get("aligned"), dict) and isinstance(qwen_result["aligned"].get("bboxes"), list):
        return qwen_result["aligned"]["bboxes"], "yxyx"
    if isinstance(qwen_result.get("bboxes"), list):
        return qwen_result["bboxes"], str(qwen_result.get("bbox_format", "xyxy"))
    return [], "xyxy"


def normalize_qwen_boxes(qwen_result: dict[str, Any], image_shape: tuple[int, ...] | list[int]) -> list[dict[str, Any]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    entries, bbox_format = _coerce_bbox_entries(qwen_result)
    force_normalized = any(key in qwen_result for key in ("parsed", "aligned"))
    detections: list[dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        box = entry.get("box_2d", entry.get("bbox", entry.get("box")))
        if box is None or len(box) != 4:
            continue
        values = [float(v) for v in box]
        normalized = force_normalized or (max(values) <= 1000.0 and min(values) >= 0.0 and max(values) > max(width, height))
        if bbox_format == "yxyx":
            ymin, xmin, ymax, xmax = values
        else:
            xmin, ymin, xmax, ymax = values
        if normalized:
            xmin = xmin / 1000.0 * width
            xmax = xmax / 1000.0 * width
            ymin = ymin / 1000.0 * height
            ymax = ymax / 1000.0 * height
        xmin = float(np.clip(xmin, 0, width - 1))
        xmax = float(np.clip(xmax, 0, width - 1))
        ymin = float(np.clip(ymin, 0, height - 1))
        ymax = float(np.clip(ymax, 0, height - 1))
        if xmin >= xmax or ymin >= ymax:
            continue
        label = str(entry.get("label", entry.get("name", f"object_{idx}"))).replace(" ", "_")
        detections.append(
            {
                "label": label,
                "bbox_xyxy": [xmin, ymin, xmax, ymax],
                "box_2d_yxyx_norm1000": [
                    int(round(ymin / height * 1000)),
                    int(round(xmin / width * 1000)),
                    int(round(ymax / height * 1000)),
                    int(round(xmax / width * 1000)),
                ],
                "source": entry,
            }
        )
    return detections


def _draw_boxes(rgb: np.ndarray, detections: list[dict[str, Any]], path: Path) -> None:
    image = Image.fromarray(_as_uint8_rgb(rgb))
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = detection["bbox_xyxy"]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1] - 14)), detection["label"], fill="red")
    image.save(path)


def _draw_masks(rgb: np.ndarray, masks: list[np.ndarray], detections: list[dict[str, Any]], path: Path) -> None:
    image = Image.fromarray(_as_uint8_rgb(rgb)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    colors = [(255, 0, 0, 90), (0, 255, 0, 90), (0, 0, 255, 90), (255, 255, 0, 90)]
    for idx, mask in enumerate(masks):
        mask_image = Image.fromarray((np.asarray(mask).astype(bool) * 255).astype(np.uint8))
        color = Image.new("RGBA", image.size, colors[idx % len(colors)])
        overlay.paste(color, (0, 0), mask_image)
    image = Image.alpha_composite(image, overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        draw.rectangle(detection["bbox_xyxy"], outline="white", width=2)
    image.save(path)


def run_sam_masks(rgb: Any, detections: list[dict[str, Any]], debug_dir: Path) -> dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    if not _env_enabled("TIPTOP_ENABLE_SAM_MASKS", False):
        result = {"success": False, "reason": "sam_disabled", "masks": [], "scores": []}
        _write_json(debug_dir / "sam_result.json", result)
        return result
    if not detections:
        result = {"success": False, "reason": "no_detections", "masks": [], "scores": []}
        _write_json(debug_dir / "sam_result.json", result)
        return result

    try:
        from tiptop.perception.sam2 import sam2_segment_objects

        rgb_array = _as_uint8_rgb(rgb)
        sam_detections = [{"box_2d": det["box_2d_yxyx_norm1000"], "label": det["label"]} for det in detections]
        raw_masks = sam2_segment_objects(Image.fromarray(rgb_array), sam_detections)
        masks_array = np.asarray(raw_masks).astype(bool)
        if masks_array.ndim == 4:
            masks = [masks_array[i, 0] for i in range(masks_array.shape[0])]
        elif masks_array.ndim == 3:
            masks = [masks_array[i] for i in range(masks_array.shape[0])]
        else:
            masks = []
        for idx, mask in enumerate(masks):
            np.save(debug_dir / f"sam_mask_{idx:02d}_{detections[idx]['label']}.npy", mask)
        _draw_masks(rgb_array, masks, detections, debug_dir / "sam_overlay.png")
        result = {"success": bool(masks), "reason": None if masks else "sam_empty_masks", "masks": masks, "scores": []}
        _write_json(debug_dir / "sam_result.json", {**result, "masks": masks})
        return result
    except Exception as exc:
        result = {
            "success": False,
            "reason": "sam_unavailable",
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "masks": [],
            "scores": [],
        }
        _write_json(debug_dir / "sam_result.json", result)
        return result


def build_object_clouds(
    xyz_map: Any,
    masks: list[np.ndarray] | np.ndarray,
    detections: list[dict[str, Any]],
    debug_dir: Path,
) -> dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir = debug_dir / "object_clouds"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz_map, dtype=np.float32)
    mask_list = list(masks) if isinstance(masks, list | tuple) else list(np.asarray(masks))
    summaries: list[dict[str, Any]] = []

    for idx, mask in enumerate(mask_list[: len(detections)]):
        mask_2d = np.asarray(mask).astype(bool)
        if mask_2d.ndim == 3:
            mask_2d = mask_2d[0]
        if mask_2d.shape != xyz.shape[:2]:
            continue
        points = xyz[mask_2d]
        valid = np.isfinite(points).all(axis=1)
        valid &= np.linalg.norm(points, axis=1) > 1e-8
        points = points[valid]
        detection = detections[idx]
        label = detection["label"]
        if len(points) > 0:
            np.save(cloud_dir / f"{idx:02d}_{label}.npy", points.astype(np.float32, copy=False))
        summary = {
            "label": label,
            "bbox": detection["bbox_xyxy"],
            "mask_area": int(mask_2d.sum()),
            "valid_xyz_count": int(len(points)),
            "centroid": points.mean(axis=0).tolist() if len(points) else None,
            "min_xyz": points.min(axis=0).tolist() if len(points) else None,
            "max_xyz": points.max(axis=0).tolist() if len(points) else None,
            "depth_min": float(points[:, 2].min()) if len(points) else None,
            "depth_max": float(points[:, 2].max()) if len(points) else None,
        }
        summaries.append(summary)

    payload = {
        "success": any(item["valid_xyz_count"] > 0 for item in summaries),
        "object_cloud_created": any(item["valid_xyz_count"] > 0 for item in summaries),
        "target_object_points": int(max([item["valid_xyz_count"] for item in summaries], default=0)),
        "objects": summaries,
    }
    _write_json(debug_dir / "object_geometry_summary.json", payload)
    return payload


def _extract_xyz_maps(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    value = observation.get("_tiptop_xyz_maps")
    if isinstance(value, dict):
        return {str(k): np.asarray(v, dtype=np.float32) for k, v in value.items()}
    return {
        key.removeprefix("xyz_"): np.asarray(value, dtype=np.float32)
        for key, value in observation.items()
        if isinstance(key, str) and key.startswith("xyz_") and hasattr(value, "shape")
    }


def run_live_perception(observation: dict[str, Any], debug_dir: str | Path) -> dict[str, Any]:
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    try:
        task_text = extract_task_text(observation)
        camera_name = select_primary_camera(observation)
        rgb = _as_uint8_rgb(observation[camera_name])
        xyz_maps = _extract_xyz_maps(observation)
        xyz_map = xyz_maps.get(camera_name)
        if xyz_map is None:
            return {
                "success": False,
                "stage": "object_cloud",
                "reason": "xyz_map_missing",
                "task_text": task_text,
                "primary_camera": camera_name,
                "debug": {"available_xyz_maps": sorted(xyz_maps.keys())},
            }

        qwen_result = run_qwen_grounding(rgb, task_text, camera_name, debug_path)
        if not qwen_result.get("success"):
            return {
                "success": False,
                "stage": "qwen",
                "reason": qwen_result.get("reason", "qwen_detection_failed"),
                "task_text": task_text,
                "primary_camera": camera_name,
                "detections": [],
                "debug": {"qwen": qwen_result},
            }

        detections = normalize_qwen_boxes(qwen_result, rgb.shape)
        _write_json(debug_path / "bbox_validation.json", {"detections": detections, "image_shape": list(rgb.shape)})
        _draw_boxes(rgb, detections, debug_path / "qwen_boxes_overlay.png")
        if not detections:
            return {
                "success": False,
                "stage": "qwen",
                "reason": "no_valid_qwen_boxes",
                "task_text": task_text,
                "primary_camera": camera_name,
                "detections": [],
                "debug": {"qwen": qwen_result},
            }

        sam_result = run_sam_masks(rgb, detections, debug_path)
        if not sam_result.get("success"):
            return {
                "success": False,
                "stage": "sam",
                "reason": sam_result.get("reason", "sam_mask_failed"),
                "task_text": task_text,
                "primary_camera": camera_name,
                "detections": detections,
                "debug": {"qwen": qwen_result, "sam": sam_result},
            }

        clouds = build_object_clouds(xyz_map, sam_result.get("masks", []), detections, debug_path)
        if not clouds.get("object_cloud_created"):
            return {
                "success": False,
                "stage": "object_cloud",
                "reason": "object_cloud_empty",
                "task_text": task_text,
                "primary_camera": camera_name,
                "detections": detections,
                "masks": sam_result.get("masks", []),
                "object_clouds": clouds,
                "debug": {"qwen": qwen_result, "sam": sam_result},
            }

        return {
            "success": True,
            "stage": "object_cloud",
            "reason": None,
            "task_text": task_text,
            "primary_camera": camera_name,
            "detections": detections,
            "masks": sam_result.get("masks", []),
            "object_clouds": clouds,
            "debug": {"qwen": qwen_result, "sam": sam_result},
        }
    except Exception as exc:
        result = {
            "success": False,
            "stage": "exception",
            "reason": str(exc),
            "debug": {"exception_type": type(exc).__name__},
        }
        _write_json(debug_path / "live_perception_failure.json", result)
        return result
