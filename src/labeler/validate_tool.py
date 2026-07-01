import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from labeler.annotation_io import (
    Annotation,
    default_annotation_path,
    read_annotations,
)
from labeler.geometry import annotation_bbox_to_xywh
from labeler.video import get_frame_count, open_video, read_frame_at


Frame = np.ndarray

WINDOW_NAME = "Annotation Validator"

FONT = int(getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0))
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize frame-level object annotations over a video."
    )
    parser.add_argument("video", type=Path, help="Path to input video file.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Path to annotation file. Defaults to <video_name>.annotations.",
    )

    return parser.parse_args(argv)


def get_annotation_for_frame(
    annotations: list[Annotation],
    frame_index: int,
) -> Annotation | None:
    """Return the annotation for a frame, or None if the annotation is missing."""
    if frame_index < 0:
        raise ValueError(f"Frame index must be non-negative, got {frame_index}")

    if frame_index >= len(annotations):
        return None

    return annotations[frame_index]


def draw_text(
    frame: Frame,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
) -> None:
    """Draw one text line on a frame."""
    cv2.putText(
        frame,
        text,
        origin,
        FONT,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_annotation_overlay(
    frame: Frame,
    annotation: Annotation | None,
) -> None:
    """Draw the frame annotation overlay in place."""
    if annotation is None:
        draw_text(frame, "UNLABELED", (10, 70), YELLOW)
        return

    if annotation.status == "V":
        x, y, width, height = annotation_bbox_to_xywh(
            annotation.x_center,
            annotation.y_center,
            annotation.width,
            annotation.height,
        )
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            GREEN,
            2,
        )
        draw_text(frame, "VISIBLE", (10, 70), GREEN)
        return

    if annotation.status == "S":
        draw_text(frame, "SKIPPED", (10, 70), YELLOW)
        return

    if annotation.status == "I":
        draw_text(frame, "INVISIBLE", (10, 70), RED)
        return

    raise ValueError(f"Unexpected annotation status: {annotation.status}")


def make_display_frame(
    frame: Frame,
    frame_index: int,
    frame_count: int,
    annotation: Annotation | None,
) -> Frame:
    """Create a display frame with annotation and controls overlaid."""
    display = frame.copy()

    draw_text(display, f"Frame {frame_index + 1} / {frame_count}", (10, 25))
    draw_text(display, "Controls: q or Esc = quit", (10, 45))
    draw_annotation_overlay(display, annotation)

    return display


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    video_path: Path = args.video
    annotation_path: Path = args.annotations or default_annotation_path(video_path)

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")

    annotations = read_annotations(annotation_path)
    capture = open_video(video_path)

    try:
        frame_count = get_frame_count(capture)
        frame = read_frame_at(capture, 0)

        if frame is None:
            raise ValueError(f"Could not read first frame from video: {video_path}")

        annotation = get_annotation_for_frame(annotations, 0)
        display = make_display_frame(
            frame=frame,
            frame_index=0,
            frame_count=frame_count,
            annotation=annotation,
        )

        cv2.imshow(WINDOW_NAME, display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in {ord("q"), 27}:  # 27 = Esc
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
