import numpy as np

from labeler.ui import GREEN, WHITE, draw_bbox, draw_text


def test_draw_text_modifies_frame() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    draw_text(frame, "hello", (10, 30), WHITE)

    assert frame.sum() > 0


def test_draw_bbox_modifies_frame() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    draw_bbox(frame, (20, 20, 40, 30), GREEN)

    assert frame.sum() > 0
