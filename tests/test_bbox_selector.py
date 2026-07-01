from labeler.bbox_selector import (
    EVENT_LBUTTONDOWN,
    EVENT_LBUTTONUP,
    EVENT_MOUSEMOVE,
    CenterDragBBoxSelector,
)


def test_selector_starts_empty() -> None:
    selector = CenterDragBBoxSelector()

    assert selector.selected_bbox() is None
    assert not selector.is_done
    assert not selector.is_dragging


def test_selector_drag_down_right() -> None:
    selector = CenterDragBBoxSelector()

    selector.handle_mouse_event(EVENT_LBUTTONDOWN, 100, 100)
    selector.handle_mouse_event(EVENT_MOUSEMOVE, 130, 120)
    selector.handle_mouse_event(EVENT_LBUTTONUP, 130, 120)

    assert selector.selected_bbox() == (70, 80, 60, 40)
    assert selector.is_done
    assert not selector.is_dragging


def test_selector_drag_up_left() -> None:
    selector = CenterDragBBoxSelector()

    selector.handle_mouse_event(EVENT_LBUTTONDOWN, 100, 100)
    selector.handle_mouse_event(EVENT_MOUSEMOVE, 70, 80)
    selector.handle_mouse_event(EVENT_LBUTTONUP, 70, 80)

    assert selector.selected_bbox() == (70, 80, 60, 40)


def test_selector_drag_up_right() -> None:
    selector = CenterDragBBoxSelector()

    selector.handle_mouse_event(EVENT_LBUTTONDOWN, 100, 100)
    selector.handle_mouse_event(EVENT_MOUSEMOVE, 130, 80)
    selector.handle_mouse_event(EVENT_LBUTTONUP, 130, 80)

    assert selector.selected_bbox() == (70, 80, 60, 40)


def test_selector_drag_down_left() -> None:
    selector = CenterDragBBoxSelector()

    selector.handle_mouse_event(EVENT_LBUTTONDOWN, 100, 100)
    selector.handle_mouse_event(EVENT_MOUSEMOVE, 70, 120)
    selector.handle_mouse_event(EVENT_LBUTTONUP, 70, 120)

    assert selector.selected_bbox() == (70, 80, 60, 40)


def test_selector_rejects_zero_size_box() -> None:
    selector = CenterDragBBoxSelector()

    selector.handle_mouse_event(EVENT_LBUTTONDOWN, 100, 100)
    selector.handle_mouse_event(EVENT_LBUTTONUP, 100, 100)

    assert selector.selected_bbox() is None
    assert not selector.is_done
    assert not selector.is_dragging


def test_selector_reset_clears_selection() -> None:
    selector = CenterDragBBoxSelector()

    selector.handle_mouse_event(EVENT_LBUTTONDOWN, 100, 100)
    selector.handle_mouse_event(EVENT_LBUTTONUP, 130, 120)

    assert selector.selected_bbox() == (70, 80, 60, 40)

    selector.reset()

    assert selector.selected_bbox() is None
    assert selector.center is None
    assert selector.current_point is None
    assert selector.bbox is None
    assert not selector.is_done
    assert not selector.is_dragging
