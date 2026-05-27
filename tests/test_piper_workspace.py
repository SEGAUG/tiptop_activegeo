from tiptop.workspace import piper_workspace


def test_piper_workspace_contains_required_obstacles():
    names = {cuboid.name for cuboid in piper_workspace()}
    assert {"desk", "rear_glass", "right_arm_keepout", "laptop_keepout"}.issubset(names)


def test_piper_workspace_obstacles_have_positive_dimensions():
    for cuboid in piper_workspace():
        assert all(dim > 0 for dim in cuboid.dims)
