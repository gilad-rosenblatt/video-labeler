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
from labeler.geometry import (
    annotation_bbox_to_xywh,
    clamp_xywh_to_frame,
    is_valid_xywh,
)
from labeler.video import get_frame_count, open_video, read_frame_at


Frame = np.ndarray

WINDOW_NAME = "Annotation Validator"

FONT = int(getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0))
LINE_AA = int(getattr(cv2, "LINE_AA", 16))

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


def clamp_frame_index(frame_index: int, frame_count: int) -> int:
    """Clamp a frame index to the valid video frame range."""
    if frame_count <= 0:
        raise ValueError(f"Frame count must be positive, got {frame_count}")

    return max(0, min(frame_index, frame_count - 1))


def next_frame_index_for_key(
    key: int,
    frame_index: int,
    frame_count: int,
) -> int | None:
    """Return next frame index for a keyboard command.

    Returns None when the user requested quit.
    """
    if key in {ord("q"), 27}:  # Esc
        return None

    if key == ord("n"):
        return clamp_frame_index(frame_index + 1, frame_count)

    if key == ord("N"):
        return clamp_frame_index(frame_index + 10, frame_count)

    if key == ord("p"):
        return clamp_frame_index(frame_index - 1, frame_count)

    if key == ord("P"):
        return clamp_frame_index(frame_index - 10, frame_count)

    return frame_index


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
        LINE_AA,
    )


def draw_annotation_overlay(
    frame: Frame,
    annotation: Annotation | None,
) -> None:
    """Draw the frame annotation overlay in place."""
    if annotation is None:
        draw_text(frame, "UNLABELED", (10, 90), YELLOW)
        return

    if annotation.status == "V":
        x, y, width, height = annotation_bbox_to_xywh(
            annotation.x_center,
            annotation.y_center,
            annotation.width,
            annotation.height,
        )

        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = clamp_xywh_to_frame(
            x=x,
            y=y,
            width=width,
            height=height,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if is_valid_xywh(x, y, width, height):
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                GREEN,
                2,
            )
            draw_text(frame, "VISIBLE", (10, 90), GREEN)
        else:
            draw_text(frame, "VISIBLE - INVALID BBOX", (10, 90), RED)

        return

    if annotation.status == "S":
        draw_text(frame, "SKIPPED", (10, 90), YELLOW)
        return

    if annotation.status == "I":
        draw_text(frame, "INVISIBLE", (10, 90), RED)
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
    draw_text(display, "Controls: n next | N +10 | p prev | P -10 | q quit", (10, 50))
    draw_text(display, "Status:", (10, 75))
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
        if frame_count <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        frame_index = 0

        while True:
            frame = read_frame_at(capture, frame_index)
            if frame is None:
                raise ValueError(f"Could not read frame {frame_index} from video: {video_path}")

            annotation = get_annotation_for_frame(annotations, frame_index)
            display = make_display_frame(
                frame=frame,
                frame_index=frame_index,
                frame_count=frame_count,
                annotation=annotation,
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(0) & 0xFF
            next_index = next_frame_index_for_key(
                key=key,
                frame_index=frame_index,
                frame_count=frame_count,
            )

            if next_index is None:
                break

            frame_index = next_index

    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
