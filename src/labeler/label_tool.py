import argparse
from pathlib import Path
from typing import Sequence

import cv2

from labeler.annotation_io import (
    Annotation,
    default_annotation_path,
    invisible_annotation,
    read_annotations,
    skipped_annotation,
    visible_annotation,
    write_annotations,
)
from labeler.bbox_selector import select_bbox_interactively
from labeler.geometry import (
    XYWH,
    clamp_xywh_to_frame,
    is_valid_xywh,
    xywh_to_annotation_bbox,
)
from labeler.ui import Frame, WHITE, draw_text
from labeler.video import get_frame_count, open_video, read_frame_at


WINDOW_NAME = "Video Labeler"

ACTION_LABEL = "label"
ACTION_SKIP = "skip"
ACTION_INVISIBLE = "invisible"
ACTION_QUIT = "quit"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively label one object of interest across a video."
    )
    parser.add_argument("video", type=Path, help="Path to input video file.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Output annotation path. Defaults to <video_name>.annotations.",
    )

    return parser.parse_args(argv)


def action_for_key(key: int) -> str | None:
    """Map keyboard input to a labeling action."""
    if key in {ord("l"), ord("L")}:
        return ACTION_LABEL

    if key in {ord("s"), ord("S")}:
        return ACTION_SKIP

    if key in {ord("i"), ord("I")}:
        return ACTION_INVISIBLE

    if key in {ord("q"), ord("Q"), 27}:  # Esc
        return ACTION_QUIT

    return None


def validate_existing_annotation_count(
    annotations: list[Annotation],
    frame_count: int,
) -> None:
    """Reject annotation files that are longer than the video."""
    if len(annotations) > frame_count:
        raise ValueError(
            f"Annotation file has {len(annotations)} annotations, "
            f"but video has only {frame_count} frames"
        )


def start_frame_index_from_existing_annotations(
    annotations: list[Annotation],
    frame_count: int,
) -> int:
    """Resume labeling after the last existing annotation."""
    validate_existing_annotation_count(annotations, frame_count)
    return len(annotations)


def visible_annotation_from_xywh(bbox: XYWH) -> Annotation:
    """Convert selected OpenCV xywh bbox into annotation format."""
    x_center, y_center, width, height = xywh_to_annotation_bbox(*bbox)
    return visible_annotation(x_center, y_center, width, height)


def make_label_display_frame(
    frame: Frame,
    frame_index: int,
    frame_count: int,
    annotation_path: Path,
) -> Frame:
    """Create display frame with labeling instructions."""
    display = frame.copy()

    draw_text(display, f"Frame {frame_index + 1} / {frame_count}", (10, 25), WHITE)
    draw_text(
        display,
        "Controls: l label | s skip | i invisible | q quit",
        (10, 50),
        WHITE,
    )
    draw_text(display, f"Output: {annotation_path}", (10, 75), WHITE)

    return display


def label_frame_manually(frame: Frame) -> Annotation | None:
    """Run manual bbox selection for one visible frame."""
    bbox = select_bbox_interactively(WINDOW_NAME, frame)

    if bbox is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    clamped_bbox = clamp_xywh_to_frame(
        x=bbox[0],
        y=bbox[1],
        width=bbox[2],
        height=bbox[3],
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if not is_valid_xywh(*clamped_bbox):
        return None

    return visible_annotation_from_xywh(clamped_bbox)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    video_path: Path = args.video
    annotation_path: Path = args.annotations or default_annotation_path(video_path)

    annotations = read_annotations(annotation_path) if annotation_path.exists() else []

    capture = open_video(video_path)

    try:
        frame_count = get_frame_count(capture)
        if frame_count <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        frame_index = start_frame_index_from_existing_annotations(
            annotations=annotations,
            frame_count=frame_count,
        )

        if frame_index == frame_count:
            print(f"All {frame_count} frames already annotated in {annotation_path}")
            return 0

        while frame_index < frame_count:
            frame = read_frame_at(capture, frame_index)
            if frame is None:
                raise ValueError(f"Could not read frame {frame_index} from video: {video_path}")

            display = make_label_display_frame(
                frame=frame,
                frame_index=frame_index,
                frame_count=frame_count,
                annotation_path=annotation_path,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(0) & 0xFF
            action = action_for_key(key)

            if action is None:
                continue

            if action == ACTION_QUIT:
                break

            if action == ACTION_SKIP:
                annotations.append(skipped_annotation())

            elif action == ACTION_INVISIBLE:
                annotations.append(invisible_annotation())

            elif action == ACTION_LABEL:
                annotation = label_frame_manually(frame)
                if annotation is None:
                    continue

                annotations.append(annotation)

            else:
                raise ValueError(f"Unexpected labeling action: {action}")

            write_annotations(annotation_path, annotations)
            frame_index += 1

    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
