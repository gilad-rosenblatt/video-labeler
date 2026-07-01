import cv2

from labeler.geometry import Point, XYWH, center_drag_to_xywh, is_valid_xywh
from labeler.ui import Frame, GREEN, RED, WHITE, YELLOW, draw_bbox, draw_text


EVENT_LBUTTONDOWN = int(getattr(cv2, "EVENT_LBUTTONDOWN", 1))
EVENT_MOUSEMOVE = int(getattr(cv2, "EVENT_MOUSEMOVE", 0))
EVENT_LBUTTONUP = int(getattr(cv2, "EVENT_LBUTTONUP", 4))

KEY_ESC = 27


class CenterDragBBoxSelector:
    """Track center-click + drag mouse interaction for manual bbox selection."""

    def __init__(self) -> None:
        self.center: Point | None = None
        self.current_point: Point | None = None
        self.bbox: XYWH | None = None
        self.is_dragging = False
        self.is_done = False

    def reset(self) -> None:
        """Clear current selection state."""
        self.center = None
        self.current_point = None
        self.bbox = None
        self.is_dragging = False
        self.is_done = False

    def handle_mouse_event(
        self,
        event: int,
        x: int,
        y: int,
        flags: int | None = None,
        param: object | None = None,
    ) -> None:
        """OpenCV mouse callback for center-click + drag selection."""
        del flags, param

        point = (x, y)

        if event == EVENT_LBUTTONDOWN:
            self.center = point
            self.current_point = point
            self.bbox = None
            self.is_dragging = True
            self.is_done = False
            return

        if event == EVENT_MOUSEMOVE and self.is_dragging and self.center is not None:
            self.current_point = point
            self.bbox = center_drag_to_xywh(self.center, point)
            return

        if event == EVENT_LBUTTONUP and self.is_dragging and self.center is not None:
            self.current_point = point
            candidate_bbox = center_drag_to_xywh(self.center, point)

            if is_valid_xywh(*candidate_bbox):
                self.bbox = candidate_bbox
                self.is_done = True
            else:
                self.bbox = None
                self.is_done = False

            self.is_dragging = False

    def selected_bbox(self) -> XYWH | None:
        """Return selected bbox after a completed valid selection."""
        if not self.is_done or self.bbox is None:
            return None

        return self.bbox


def draw_selector_overlay(
    frame: Frame,
    selector: CenterDragBBoxSelector,
) -> None:
    """Draw selector instructions and current preview bbox."""
    draw_text(frame, "Click object center, drag to bbox edge, release", (10, 25), WHITE)
    draw_text(frame, "Esc/q = cancel", (10, 50), WHITE)

    if selector.center is not None:
        cv2.circle(frame, selector.center, 3, YELLOW, -1)

    if selector.bbox is not None and is_valid_xywh(*selector.bbox):
        draw_bbox(frame, selector.bbox, GREEN)

    if selector.center is not None and selector.bbox is None:
        draw_text(frame, "Drag outward to create a non-empty box", (10, 75), RED)


def select_bbox_interactively(
    window_name: str,
    frame: Frame,
) -> XYWH | None:
    """Let the user manually select one bbox.

    Returns:
        xywh bbox if selected, or None if canceled.
    """
    selector = CenterDragBBoxSelector()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, selector.handle_mouse_event)

    while True:
        display = frame.copy()
        draw_selector_overlay(display, selector)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        if key in {ord("q"), KEY_ESC}:
            return None

        bbox = selector.selected_bbox()
        if bbox is not None:
            return bbox
