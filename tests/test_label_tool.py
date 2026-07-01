from pathlib import Path

import numpy as np
import pytest

from labeler.annotation_io import skipped_annotation, visible_annotation
from labeler.label_tool import (
    ACTION_ACCEPT,
    ACTION_FIX,
    ACTION_INVISIBLE,
    ACTION_LABEL,
    ACTION_QUIT,
    ACTION_SKIP,
    action_for_key,
    clamp_bbox_to_frame,
    make_label_display_frame,
    start_frame_index_from_existing_annotations,
    validate_existing_annotation_count,
    visible_annotation_from_xywh,
)


def test_action_for_key_label_without_prediction() -> None:
    assert action_for_key(ord("l"), has_prediction=False) == ACTION_LABEL
    assert action_for_key(ord("L"), has_prediction=False) == ACTION_LABEL


def test_action_for_key_label_ignored_with_prediction() -> None:
    assert action_for_key(ord("l"), has_prediction=True) is None
    assert action_for_key(ord("L"), has_prediction=True) is None


def test_action_for_key_accept_with_prediction() -> None:
    assert action_for_key(ord("a"), has_prediction=True) == ACTION_ACCEPT
    assert action_for_key(ord("A"), has_prediction=True) == ACTION_ACCEPT


def test_action_for_key_accept_ignored_without_prediction() -> None:
    assert action_for_key(ord("a"), has_prediction=False) is None
    assert action_for_key(ord("A"), has_prediction=False) is None


def test_action_for_key_fix_with_prediction() -> None:
    assert action_for_key(ord("f"), has_prediction=True) == ACTION_FIX
    assert action_for_key(ord("F"), has_prediction=True) == ACTION_FIX


def test_action_for_key_fix_ignored_without_prediction() -> None:
    assert action_for_key(ord("f"), has_prediction=False) is None
    assert action_for_key(ord("F"), has_prediction=False) is None


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


def test_clamp_bbox_to_frame_when_valid() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    bbox = clamp_bbox_to_frame(frame, (10, 20, 30, 40))

    assert bbox == (10, 20, 30, 40)


def test_clamp_bbox_to_frame_when_partly_outside() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    bbox = clamp_bbox_to_frame(frame, (-10, -20, 50, 60))

    assert bbox == (0, 0, 40, 40)


def test_clamp_bbox_to_frame_returns_none_when_invalid_after_clamp() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    bbox = clamp_bbox_to_frame(frame, (200, 200, 30, 40))

    assert bbox is None


def test_make_label_display_frame_without_prediction() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    display = make_label_display_frame(
        frame=frame,
        frame_index=0,
        frame_count=10,
        annotation_path=Path("sample.annotations"),
    )

    assert display.shape == frame.shape
    assert display.sum() > 0


def test_make_label_display_frame_with_prediction() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    display = make_label_display_frame(
        frame=frame,
        frame_index=0,
        frame_count=10,
        annotation_path=Path("sample.annotations"),
        predicted_bbox=(20, 20, 40, 30),
    )

    assert display.shape == frame.shape
    assert display.sum() > 0
