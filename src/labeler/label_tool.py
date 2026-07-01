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
from labeler.tracker import OpenCVObjectTracker
from labeler.ui import Frame, WHITE, YELLOW, draw_bbox, draw_text
from labeler.video import get_frame_count, open_video, read_frame_at


WINDOW_NAME = "Video Labeler"

ACTION_LABEL = "label"
ACTION_ACCEPT = "accept"
ACTION_FIX = "fix"
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


def action_for_key(key: int, has_prediction: bool = False) -> str | None:
    """Map keyboard input to a labeling action."""
    if has_prediction:
        if key in {ord("a"), ord("A")}:
            return ACTION_ACCEPT

        if key in {ord("f"), ord("F")}:
            return ACTION_FIX

    else:
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


def clamp_bbox_to_frame(frame: Frame, bbox: XYWH) -> XYWH | None:
    """Clamp a bbox to the frame, returning None if it becomes invalid."""
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

    return clamped_bbox


def make_label_display_frame(
    frame: Frame,
    frame_index: int,
    frame_count: int,
    annotation_path: Path,
    predicted_bbox: XYWH | None = None,
) -> Frame:
    """Create display frame with labeling instructions."""
    display = frame.copy()

    draw_text(display, f"Frame {frame_index + 1} / {frame_count}", (10, 25), WHITE)

    if predicted_bbox is None:
        controls = "Controls: l label | s skip | i invisible | q quit"
    else:
        controls = "Controls: a accept | f fix | s skip | i invisible | q quit"
        draw_bbox(display, predicted_bbox, YELLOW)
        draw_text(display, "Tracker prediction", (10, 100), YELLOW)

    draw_text(display, controls, (10, 50), WHITE)
    draw_text(display, f"Output: {annotation_path}", (10, 75), WHITE)

    return display


def select_manual_bbox(frame: Frame) -> XYWH | None:
    """Run manual center-drag bbox selection and clamp result to the frame."""
    bbox = select_bbox_interactively(WINDOW_NAME, frame)

    if bbox is None:
        return None

    return clamp_bbox_to_frame(frame, bbox)


def initialize_tracker_safely(
    tracker: OpenCVObjectTracker,
    frame: Frame,
    bbox: XYWH,
) -> None:
    """Initialize tracker, but keep manual labeling usable if tracker init fails."""
    try:
        tracker.initialize(frame, bbox)
    except RuntimeError as exc:
        print(f"Warning: tracker could not be initialized: {exc}")
        tracker.reset()


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    video_path: Path = args.video
    annotation_path: Path = args.annotations or default_annotation_path(video_path)

    annotations = read_annotations(annotation_path) if annotation_path.exists() else []

    capture = open_video(video_path)
    tracker = OpenCVObjectTracker()

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

            predicted_bbox = tracker.update(frame) if tracker.has_tracker() else None

            display = make_label_display_frame(
                frame=frame,
                frame_index=frame_index,
                frame_count=frame_count,
                annotation_path=annotation_path,
                predicted_bbox=predicted_bbox,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(0) & 0xFF
            action = action_for_key(key, has_prediction=predicted_bbox is not None)

            if action is None:
                continue

            if action == ACTION_QUIT:
                break

            if action == ACTION_SKIP:
                annotations.append(skipped_annotation())
                tracker.reset()

            elif action == ACTION_INVISIBLE:
                annotations.append(invisible_annotation())
                tracker.reset()

            elif action == ACTION_ACCEPT:
                if predicted_bbox is None:
                    continue

                bbox = clamp_bbox_to_frame(frame, predicted_bbox)
                if bbox is None:
                    tracker.reset()
                    continue

                annotations.append(visible_annotation_from_xywh(bbox))
                initialize_tracker_safely(tracker, frame, bbox)

            elif action in {ACTION_LABEL, ACTION_FIX}:
                bbox = select_manual_bbox(frame)
                if bbox is None:
                    continue

                annotations.append(visible_annotation_from_xywh(bbox))
                initialize_tracker_safely(tracker, frame, bbox)

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
