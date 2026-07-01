import numpy as np
import pytest

from labeler.annotation_io import (
    invisible_annotation,
    skipped_annotation,
    visible_annotation,
)
from labeler.validate_tool import (
    clamp_frame_index,
    get_annotation_for_frame,
    make_display_frame,
    next_frame_index_for_key,
)


def test_get_annotation_for_frame_returns_annotation() -> None:
    annotations = [
        visible_annotation(20, 20, 10, 10),
        skipped_annotation(),
    ]

    annotation = get_annotation_for_frame(annotations, 1)

    assert annotation == skipped_annotation()


def test_get_annotation_for_frame_returns_none_when_missing() -> None:
    annotations = [visible_annotation(20, 20, 10, 10)]

    annotation = get_annotation_for_frame(annotations, 5)

    assert annotation is None


def test_get_annotation_for_frame_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        get_annotation_for_frame([], -1)


def test_clamp_frame_index() -> None:
    assert clamp_frame_index(0, frame_count=10) == 0
    assert clamp_frame_index(5, frame_count=10) == 5
    assert clamp_frame_index(-5, frame_count=10) == 0
    assert clamp_frame_index(100, frame_count=10) == 9


def test_clamp_frame_index_rejects_empty_video() -> None:
    with pytest.raises(ValueError, match="Frame count must be positive"):
        clamp_frame_index(0, frame_count=0)


def test_next_frame_index_for_key_next() -> None:
    assert next_frame_index_for_key(ord("n"), frame_index=3, frame_count=10) == 4


def test_next_frame_index_for_key_jump_forward() -> None:
    assert next_frame_index_for_key(ord("N"), frame_index=3, frame_count=20) == 13


def test_next_frame_index_for_key_previous() -> None:
    assert next_frame_index_for_key(ord("p"), frame_index=3, frame_count=10) == 2


def test_next_frame_index_for_key_jump_backward() -> None:
    assert next_frame_index_for_key(ord("P"), frame_index=13, frame_count=20) == 3


def test_next_frame_index_for_key_clamps_forward() -> None:
    assert next_frame_index_for_key(ord("N"), frame_index=8, frame_count=10) == 9


def test_next_frame_index_for_key_clamps_backward() -> None:
    assert next_frame_index_for_key(ord("P"), frame_index=3, frame_count=10) == 0


def test_next_frame_index_for_key_quit() -> None:
    assert next_frame_index_for_key(ord("q"), frame_index=3, frame_count=10) is None


def test_next_frame_index_for_key_escape() -> None:
    assert next_frame_index_for_key(27, frame_index=3, frame_count=10) is None


def test_next_frame_index_for_key_ignores_unknown_key() -> None:
    assert next_frame_index_for_key(ord("x"), frame_index=3, frame_count=10) == 3


def test_make_display_frame_visible_annotation() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    annotation = visible_annotation(50, 40, 20, 10)

    display = make_display_frame(
        frame=frame,
        frame_index=0,
        frame_count=10,
        annotation=annotation,
    )

    assert display.shape == frame.shape
    assert display.sum() > 0


def test_make_display_frame_skipped_annotation() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    display = make_display_frame(
        frame=frame,
        frame_index=0,
        frame_count=10,
        annotation=skipped_annotation(),
    )

    assert display.shape == frame.shape
    assert display.sum() > 0


def test_make_display_frame_invisible_annotation() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    display = make_display_frame(
        frame=frame,
        frame_index=0,
        frame_count=10,
        annotation=invisible_annotation(),
    )

    assert display.shape == frame.shape
    assert display.sum() > 0


def test_make_display_frame_missing_annotation() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    display = make_display_frame(
        frame=frame,
        frame_index=0,
        frame_count=10,
        annotation=None,
    )

    assert display.shape == frame.shape
    assert display.sum() > 0
