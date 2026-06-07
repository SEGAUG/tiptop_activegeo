import json

import numpy as np

from tiptop.scripts.piper_qwen_m2t2_dry_run import (
    discover_manual_multiview_h5_paths,
    filter_grasps_by_detection_masks,
    load_qwen_aligned_detection,
    resolve_multiview_h5_paths,
)


def test_load_qwen_aligned_detection_validates_required_keys(tmp_path):
    path = tmp_path / "gemini_aligned.json"
    path.write_text(
        json.dumps({
            "bboxes": [{"box_2d": [73, 570, 518, 752], "label": "blue_bottle"}],
            "grounded_atoms": [{"predicate": "holding", "args": ["blue_bottle"]}],
        }),
        encoding="utf-8",
    )

    detection = load_qwen_aligned_detection(path)

    assert detection["bboxes"][0]["box_2d"] == [73, 570, 518, 752]
    assert detection["grounded_atoms"] == [{"predicate": "holding", "args": ["blue_bottle"]}]


def test_discover_manual_multiview_h5_paths_returns_sample_order(tmp_path):
    run_dir = tmp_path / "manual_run"
    for sample_name in ["sample_010", "sample_002", "sample_000"]:
        sample_dir = run_dir / sample_name
        sample_dir.mkdir(parents=True)
        (sample_dir / "observation_calibrated.h5").write_bytes(b"not real h5")

    paths = discover_manual_multiview_h5_paths(run_dir)

    assert [path.parent.name for path in paths] == ["sample_000", "sample_002", "sample_010"]


def test_resolve_multiview_h5_paths_combines_run_and_explicit_paths(tmp_path):
    run_dir = tmp_path / "manual_run"
    sample_dir = run_dir / "sample_000"
    sample_dir.mkdir(parents=True)
    run_h5 = sample_dir / "observation_calibrated.h5"
    explicit_h5 = tmp_path / "extra.h5"
    run_h5.write_bytes(b"not real h5")
    explicit_h5.write_bytes(b"not real h5")

    paths = resolve_multiview_h5_paths(str(run_dir), str(explicit_h5))

    assert paths == [run_h5, explicit_h5]


def test_filter_grasps_by_detection_masks_associates_contacts_to_detected_objects():
    xyz_map = np.zeros((3, 3, 3), dtype=np.float32)
    xyz_map[0, 0] = [0.0, 0.0, 0.0]
    xyz_map[0, 1] = [0.01, 0.0, 0.0]
    xyz_map[2, 2] = [1.0, 0.0, 0.0]

    masks = np.zeros((2, 1, 3, 3), dtype=bool)
    masks[0, 0, 0, 0] = True
    masks[0, 0, 0, 1] = True
    masks[1, 0, 2, 2] = True
    bboxes = [{"label": "bottle"}, {"label": "ball"}]

    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    grasps = {
        "object_0": {
            "poses": poses,
            "confidences": np.array([0.9, 0.8, 0.7]),
            "contacts": np.array([[0.001, 0.0, 0.0], [1.001, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        }
    }

    filtered = filter_grasps_by_detection_masks(xyz_map, masks, bboxes, grasps, contact_threshold=0.02)

    assert set(filtered) == {"bottle", "ball"}
    assert len(filtered["bottle"]["poses"]) == 1
    assert len(filtered["ball"]["poses"]) == 1
    assert filtered["bottle"]["confidences"].tolist() == [0.9]
    assert filtered["ball"]["confidences"].tolist() == [0.8]
