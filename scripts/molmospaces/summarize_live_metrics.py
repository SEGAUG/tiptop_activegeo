#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_reasons = Counter(str(row.get("fallback_reason")) for row in rows if row.get("fallback_used"))
    latest = rows[-1] if rows else {}
    return {
        "episodes": len(rows),
        "planning_attempted": sum(1 for row in rows if row.get("planning_attempted")),
        "planning_success": sum(1 for row in rows if row.get("planning_success")),
        "fallback_used": sum(1 for row in rows if row.get("fallback_used")),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "first_non_hold_action_steps": [
            row.get("first_non_hold_action_step")
            for row in rows
            if row.get("first_non_hold_action_step") is not None
        ],
        "backends": sorted({str(row.get("backend")) for row in rows if row.get("backend") is not None}),
        "live_planning_enabled": any(bool(row.get("live_planning_enabled")) for row in rows),
        "require_plan": any(bool(row.get("require_plan")) for row in rows),
        "depth_available": sum(1 for row in rows if row.get("depth_available")),
        "xyz_map_created": sum(1 for row in rows if row.get("xyz_map_created")),
        "depth_backends": sorted({str(row.get("depth_backend")) for row in rows if row.get("depth_backend") is not None}),
        "latest": latest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TiPToP MolmoSpaces live policy metrics.")
    parser.add_argument(
        "metrics_path",
        nargs="?",
        default="logs/molmospaces_live/episode_metrics.jsonl",
        help="Path to episode_metrics.jsonl",
    )
    args = parser.parse_args()
    path = Path(args.metrics_path)
    print(json.dumps(summarize(load_metrics(path)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
