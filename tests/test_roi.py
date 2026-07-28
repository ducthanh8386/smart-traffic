from core.roi import create_default_roi


def test_full_frame_roi_clamps_to_frame_bounds() -> None:
    roi = create_default_roi(640, 480, {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})

    assert roi.tolist() == [[0, 0], [639, 0], [639, 479], [0, 479]]
