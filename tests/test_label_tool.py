from pathlib import Path

import numpy as np
import pytest

from labeler.annotation_io import skipped_annotation, visible_annotation
from labeler.label_tool import (
    ACTION_INVISIBLE,
    ACTION_LABEL,
    ACTION_QUIT,
    ACTION_SKIP,
    action_for_key,
    make_label_display_frame,
    start_frame_index_from_existing_annotations,
    validate_existing_annotation_count,
    visible_annotation_from_xywh,
)


def test_action_for_key_label() -> None:
    assert action_for_key(ord("l")) == ACTION_LABEL
    assert action_for_key(ord("L")) == ACTION_LABEL


def test_action_for_key_skip() -> None:
    assert action_for_key(ord("s")) == ACTION_SKIP
    assert action_for_key(ord("S")) == ACTION_SKIP


def test_action_for_key_invisible() -> None:
    assert action_for_key(ord("i")) == ACTION_INVISIBLE
    assert action_for_key(ord("I")) == ACTION_INVISIBLE


def test_action_for_key_quit() -> None:
    assert action_for_key(ord("q")) == ACTION_QUIT
    assert action_for_key(ord("Q")) == ACTION_QUIT
    assert action_for_key(27) == ACTION_QUIT


def test_action_for_key_unknown() -> None:
    assert action_for_key(ord("x")) is None


def test_validate_existing_annotation_count_accepts_shorter_file() -> None:
    annotations = [skipped_annotation()]

    validate_existing_annotation_count(annotations, frame_count=3)


def test_validate_existing_annotation_count_accepts_equal_count() -> None:
    annotations = [skipped_annotation(), skipped_annotation()]

    validate_existing_annotation_count(annotations, frame_count=2)


def test_validate_existing_annotation_count_rejects_longer_file() -> None:
    annotations = [skipped_annotation(), skipped_annotation(), skipped_annotation()]

    with pytest.raises(ValueError, match="Annotation file has 3 annotations"):
        validate_existing_annotation_count(annotations, frame_count=2)


def test_start_frame_index_from_existing_annotations() -> None:
    annotations = [skipped_annotation(), skipped_annotation()]

    assert start_frame_index_from_existing_annotations(annotations, frame_count=5) == 2


def test_visible_annotation_from_xywh() -> None:
    annotation = visible_annotation_from_xywh((70, 80, 60, 40))

    assert annotation == visible_annotation(100, 100, 60, 40)


def test_make_label_display_frame() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    display = make_label_display_frame(
        frame=frame,
        frame_index=0,
        frame_count=10,
        annotation_path=Path("sample.annotations"),
    )

    assert display.shape == frame.shape
    assert display.sum() > 0
