from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from tiptop.perception.gemini import load_json, load_prompt

DASHSCOPE_CHAT_COMPLETIONS_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def align_qwen_detection_to_gemini(
    parsed: dict[str, Any], *, qwen_bbox_format: str = "xyxy"
) -> tuple[list[dict], list[dict]]:
    """Convert Qwen detection JSON to TiPToP's Gemini tuple format.

    Qwen VL returns normalized bboxes as [xmin, ymin, xmax, ymax] in our tests,
    while TiPToP/SAM2 expects [ymin, xmin, ymax, xmax].
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
    return bboxes, grounded_atoms


def aligned_detection_dict(parsed: dict[str, Any], *, qwen_bbox_format: str = "xyxy") -> dict[str, Any]:
    bboxes, grounded_atoms = align_qwen_detection_to_gemini(parsed, qwen_bbox_format=qwen_bbox_format)
    return {"bboxes": bboxes, "grounded_atoms": grounded_atoms}


def _image_data_url(rgb_pil: Image.Image) -> str:
    buffer = io.BytesIO()
    rgb_pil.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def _clean_api_key(api_key: str | None, *, env_name: str) -> str:
    if not api_key:
        raise RuntimeError(f"{env_name} is not set")
    if "\n" in api_key or " " in api_key:
        raise RuntimeError(f"{env_name} contains whitespace; re-export it as a clean API key")
    return api_key


def _open_dashscope_request(request: urllib.request.Request, *, timeout_s: float):
    # Avoid inheriting lab HTTP(S)_PROXY settings here. We only need a direct
    # HTTPS call to DashScope, and proxy tunnels have been observed to reset.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout_s)


def qwen_detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    *,
    api_key: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    model_id: str | None = None,
    qwen_bbox_format: str = "xyxy",
    timeout_s: float = 90.0,
    max_tokens: int = 2048,
) -> tuple[list[dict], list[dict]]:
    """Detect objects and translate task with DashScope Qwen-VL.

    Returns the same tuple shape as Gemini: (bboxes, grounded_atoms).
    """
    api_key = _clean_api_key(api_key or os.environ.get(api_key_env), env_name=api_key_env)
    model_id = model_id or os.environ.get("QWEN_VL_MODEL", "qwen3-vl-flash")

    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    prompt += """

Return ONLY valid JSON, no markdown, no explanation.
Use exactly:
{
  "bboxes": [{"box_2d": [xmin, ymin, xmax, ymax], "label": "object_name"}],
  "predicates": [{"name": "holding", "args": ["object_name"]}]
}
box_2d must be integers normalized 0-1000 in [xmin, ymin, xmax, ymax] order.
Labels must be English snake_case.
Predicate args must exactly match detected labels.
""".strip()

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        DASHSCOPE_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _open_dashscope_request(request, timeout_s=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen VL request failed with HTTP {exc.code}: {error_body}") from exc

    raw_text = body["choices"][0]["message"]["content"]
    parsed = load_json(raw_text)
    return align_qwen_detection_to_gemini(parsed, qwen_bbox_format=qwen_bbox_format)


def qwen_detect_and_translate_raw(
    image: Image.Image,
    task_instruction: str,
    *,
    api_key: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    model_id: str | None = None,
    qwen_bbox_format: str = "xyxy",
    timeout_s: float = 90.0,
    max_tokens: int = 2048,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Debug variant returning raw text, parsed Qwen JSON, and Gemini-aligned dict."""
    api_key = _clean_api_key(api_key or os.environ.get(api_key_env), env_name=api_key_env)
    model_id = model_id or os.environ.get("QWEN_VL_MODEL", "qwen3-vl-flash")

    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    prompt += """

Return ONLY valid JSON, no markdown, no explanation.
Use exactly:
{
  "bboxes": [{"box_2d": [xmin, ymin, xmax, ymax], "label": "object_name"}],
  "predicates": [{"name": "holding", "args": ["object_name"]}]
}
box_2d must be integers normalized 0-1000 in [xmin, ymin, xmax, ymax] order.
Labels must be English snake_case.
Predicate args must exactly match detected labels.
""".strip()

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        DASHSCOPE_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _open_dashscope_request(request, timeout_s=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen VL request failed with HTTP {exc.code}: {error_body}") from exc

    raw_text = body["choices"][0]["message"]["content"]
    parsed = load_json(raw_text)
    aligned = aligned_detection_dict(parsed, qwen_bbox_format=qwen_bbox_format)
    return raw_text, parsed, aligned
