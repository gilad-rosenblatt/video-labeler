from skyways_cv_labeler.geometry import (
    annotation_bbox_to_xywh,
    center_drag_to_xywh,
    clamp_xywh_to_frame,
    is_valid_xywh,
    xywh_to_annotation_bbox,
)


def test_center_drag_to_xywh_drag_down_right() -> None:
    bbox = center_drag_to_xywh(center=(100, 100), drag_point=(130, 120))

    assert bbox == (70, 80, 60, 40)


def test_center_drag_to_xywh_drag_up_left() -> None:
    bbox = center_drag_to_xywh(center=(100, 100), drag_point=(70, 80))

    assert bbox == (70, 80, 60, 40)


def test_center_drag_to_xywh_drag_up_right() -> None:
    bbox = center_drag_to_xywh(center=(100, 100), drag_point=(130, 80))

    assert bbox == (70, 80, 60, 40)


def test_center_drag_to_xywh_drag_down_left() -> None:
    bbox = center_drag_to_xywh(center=(100, 100), drag_point=(70, 120))

    assert bbox == (70, 80, 60, 40)


def test_xywh_to_annotation_bbox() -> None:
    annotation_bbox = xywh_to_annotation_bbox(x=70, y=80, width=60, height=40)

    assert annotation_bbox == (100, 100, 60, 40)


def test_annotation_bbox_to_xywh() -> None:
    xywh = annotation_bbox_to_xywh(x_center=100, y_center=100, width=60, height=40)

    assert xywh == (70, 80, 60, 40)


def test_xywh_annotation_round_trip() -> None:
    original_xywh = (70, 80, 60, 40)

    annotation_bbox = xywh_to_annotation_bbox(*original_xywh)
    round_trip_xywh = annotation_bbox_to_xywh(*annotation_bbox)

    assert round_trip_xywh == original_xywh


def test_clamp_xywh_to_frame_when_inside_frame() -> None:
    bbox = clamp_xywh_to_frame(
        x=10,
        y=20,
        width=30,
        height=40,
        frame_width=100,
        frame_height=100,
    )

    assert bbox == (10, 20, 30, 40)


def test_clamp_xywh_to_frame_when_partly_outside_top_left() -> None:
    bbox = clamp_xywh_to_frame(
        x=-10,
        y=-20,
        width=50,
        height=60,
        frame_width=100,
        frame_height=100,
    )

    assert bbox == (0, 0, 40, 40)


def test_clamp_xywh_to_frame_when_partly_outside_bottom_right() -> None:
    bbox = clamp_xywh_to_frame(
        x=80,
        y=70,
        width=40,
        height=50,
        frame_width=100,
        frame_height=100,
    )

    assert bbox == (80, 70, 20, 30)


def test_clamp_xywh_to_frame_when_fully_outside() -> None:
    bbox = clamp_xywh_to_frame(
        x=120,
        y=130,
        width=40,
        height=50,
        frame_width=100,
        frame_height=100,
    )

    assert bbox == (120, 130, 0, 0)


def test_is_valid_xywh() -> None:
    assert is_valid_xywh(0, 0, 10, 20)
    assert not is_valid_xywh(-1, 0, 10, 20)
    assert not is_valid_xywh(0, -1, 10, 20)
    assert not is_valid_xywh(0, 0, 0, 20)
    assert not is_valid_xywh(0, 0, 10, 0)
