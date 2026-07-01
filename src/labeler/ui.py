import cv2
import numpy as np

from labeler.geometry import XYWH


Frame = np.ndarray
Color = tuple[int, int, int]
Point = tuple[int, int]

FONT = int(getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0))
LINE_AA = int(getattr(cv2, "LINE_AA", 16))

WHITE: Color = (255, 255, 255)
GREEN: Color = (0, 255, 0)
YELLOW: Color = (0, 255, 255)
RED: Color = (0, 0, 255)


def draw_text(
    frame: Frame,
    text: str,
    origin: Point,
    color: Color = WHITE,
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """Draw one text line on a frame."""
    cv2.putText(
        frame,
        text,
        origin,
        FONT,
        scale,
        color,
        thickness,
        LINE_AA,
    )


def draw_bbox(
    frame: Frame,
    bbox: XYWH,
    color: Color = GREEN,
    thickness: int = 2,
) -> None:
    """Draw an OpenCV xywh bounding box on a frame."""
    x, y, width, height = bbox

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        color,
        thickness,
    )
