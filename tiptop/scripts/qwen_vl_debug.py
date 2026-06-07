from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tyro
from PIL import Image

from tiptop.perception.qwen_vl import qwen_detect_and_translate_raw
from tiptop.piper import PiperRobotClient

def align_qwen_detection_to_gemini(parsed: dict[str, Any], *, qwen_bbox_format: str = "xyxy") -> dict[str, Any]:
    """Convert Qwen detection JSON to TiPToP's Gemini-aligned detection structure.

    Qwen VL currently returns normalized bboxes as [xmin, ymin, xmax, ymax] in our tests,
    while the TiPToP/SAM2 path expects [ymin, xmin, ymax, xmax].
    """
    if qwen_bbox_format not in {"xyxy", "yxyx"}:
        raise ValueError(f"qwen_bbox_format must be 'xyxy' or 'yxyx', got {qwen_bbox_format!r}")

    bboxes = []
    for bbox in parsed.get("bboxes", []):
        box_2d = bbox.get("box_2d", [])
        if len(box_2d) != 4:
            continue
        a, b, c, d = [int(round(float(v))) for v in box_2d]
        if qwen_bbox_format == "xyxy":
            xmin, ymin, xmax, ymax = a, b, c, d
            converted_box = [ymin, xmin, ymax, xmax]
        else:
            converted_box = [a, b, c, d]
        bboxes.append({**bbox, "box_2d": converted_box})

    grounded_atoms = [
        {"predicate": predicate["name"], "args": predicate["args"]}
        for predicate in parsed.get("predicates", [])
        if predicate.get("name") and predicate.get("args")
    ]
    return {"bboxes": bboxes, "grounded_atoms": grounded_atoms}


def _draw_bboxes(
    rgb: np.ndarray,
    qwen_bboxes: list[dict[str, Any]],
    aligned_bboxes: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    for bbox in qwen_bboxes:
        box = bbox.get("box_2d", [])
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        x1, y1 = int(xmin / 1000 * width), int(ymin / 1000 * height)
        x2, y2 = int(xmax / 1000 * width), int(ymax / 1000 * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            "qwen_xyxy " + str(bbox.get("label", "")),
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    for bbox in aligned_bboxes:
        box = bbox.get("box_2d", [])
        if len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = box
        x1, y1 = int(xmin / 1000 * width), int(ymin / 1000 * height)
        x2, y2 = int(xmax / 1000 * width), int(ymax / 1000 * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            image,
            "tiptop_yxyx",
            (x1, max(48, y1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(output_path), image)


def qwen_vl_debug(
    bridge_url: str | None = None,
    task_instruction: str = "Pick up the bottle.",
    output_dir: str = "/data/data2/jinhui.lin/code/aicode/piper_real_outputs/qwen_vl_debug",
    model: str = "qwen3-vl-flash",
    api_key_env: str = "DASHSCOPE_API_KEY",
    qwen_bbox_format: str = "xyxy",
    timeout_s: float = 90.0,
    max_tokens: int = 2048,
) -> None:
    """Capture one Piper image, call Qwen VL, and save Gemini-aligned detection outputs."""
    bridge_url = bridge_url or os.environ.get("TIPTOP_PIPER_BRIDGE_URL", "http://127.0.0.1:8766")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    if "\n" in api_key or " " in api_key:
        raise RuntimeError(f"{api_key_env} contains whitespace; re-export it as a clean API key")

    client = PiperRobotClient(base_url=bridge_url, timeout_s=10.0)
    snapshot = client.get_snapshot()
    rgb = snapshot["rgb"].astype(np.uint8)
    rgb_pil = Image.fromarray(rgb)

    raw_text, parsed, aligned = qwen_detect_and_translate_raw(
        rgb_pil,
        task_instruction,
        api_key=api_key,
        model_id=model,
        qwen_bbox_format=qwen_bbox_format,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )

    save_dir = Path(output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (save_dir / "qwen_raw_text.txt").write_text(raw_text, encoding="utf-8")
    (save_dir / "qwen_parsed.json").write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    (save_dir / "gemini_aligned.json").write_text(json.dumps(aligned, indent=2, ensure_ascii=False), encoding="utf-8")
    _draw_bboxes(rgb, parsed.get("bboxes", []), aligned.get("bboxes", []), save_dir / "qwen_bbox_viz.png")

    print("===== Qwen raw text =====")
    print(raw_text)
    print("\n===== Gemini-aligned output =====")
    print(json.dumps(aligned, indent=2, ensure_ascii=False))
    print("\nsaved:", save_dir)


def entrypoint() -> None:
    tyro.cli(qwen_vl_debug)


if __name__ == "__main__":
    entrypoint()
