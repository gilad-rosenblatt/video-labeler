Point = tuple[int, int]
XYWH = tuple[int, int, int, int]
AnnotationBBox = tuple[int, int, int, int]


def center_drag_to_xywh(center: Point, drag_point: Point) -> XYWH:
    """Convert center-click + drag-point interaction to OpenCV xywh bbox.

    The user clicks the object center and drags outward to define half-width
    and half-height.

    Returns:
        x, y, width, height
    """
    center_x, center_y = center
    drag_x, drag_y = drag_point

    half_width = abs(drag_x - center_x)
    half_height = abs(drag_y - center_y)

    x = center_x - half_width
    y = center_y - half_height
    width = 2 * half_width
    height = 2 * half_height

    return x, y, width, height


def xywh_to_annotation_bbox(x: int, y: int, width: int, height: int) -> AnnotationBBox:
    """Convert OpenCV xywh bbox to annotation bbox.

    OpenCV/tracker format:
        x, y, width, height

    Annotation format:
        x_center, y_center, width, height
    """
    x_center = x + width // 2
    y_center = y + height // 2

    return x_center, y_center, width, height


def annotation_bbox_to_xywh(
    x_center: int,
    y_center: int,
    width: int,
    height: int,
) -> XYWH:
    """Convert annotation bbox to OpenCV xywh bbox."""
    x = x_center - width // 2
    y = y_center - height // 2

    return x, y, width, height


def clamp_xywh_to_frame(
    x: int,
    y: int,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
) -> XYWH:
    """Clamp an xywh bbox so it stays inside frame bounds."""
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + width)
    y2 = min(frame_height, y + height)

    clamped_width = max(0, x2 - x1)
    clamped_height = max(0, y2 - y1)

    return x1, y1, clamped_width, clamped_height


def is_valid_xywh(x: int, y: int, width: int, height: int) -> bool:
    """Return True if bbox has non-negative origin and positive size."""
    return x >= 0 and y >= 0 and width > 0 and height > 0
