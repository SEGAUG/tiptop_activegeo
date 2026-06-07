from tiptop.scripts.qwen_vl_debug import align_qwen_detection_to_gemini


def test_align_qwen_xyxy_bboxes_to_gemini_yxyx():
    parsed = {
        "bboxes": [
            {"box_2d": [570, 73, 752, 518], "label": "blue_bottle_with_palm_tree"},
        ],
        "predicates": [
            {"name": "holding", "args": ["blue_bottle_with_palm_tree"]},
        ],
    }

    aligned = align_qwen_detection_to_gemini(parsed)

    assert aligned == {
        "bboxes": [
            {"box_2d": [73, 570, 518, 752], "label": "blue_bottle_with_palm_tree"},
        ],
        "grounded_atoms": [
            {"predicate": "holding", "args": ["blue_bottle_with_palm_tree"]},
        ],
    }
