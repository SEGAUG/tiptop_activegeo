from tiptop.perception.qwen_vl import align_qwen_detection_to_gemini


def test_qwen_backend_aligns_xyxy_to_tiptop_yxyx():
    parsed = {
        "bboxes": [{"box_2d": [570, 73, 752, 518], "label": "blue_bottle"}],
        "predicates": [{"name": "holding", "args": ["blue_bottle"]}],
    }

    bboxes, grounded_atoms = align_qwen_detection_to_gemini(parsed)

    assert bboxes == [{"box_2d": [73, 570, 518, 752], "label": "blue_bottle"}]
    assert grounded_atoms == [{"predicate": "holding", "args": ["blue_bottle"]}]
