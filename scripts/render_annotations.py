import argparse
from pathlib import Path
from typing import Sequence

import cv2

from labeler.annotation_io import Annotation, default_annotation_path, read_annotations
from labeler.geometry import annotation_bbox_to_xywh, clamp_xywh_to_frame, is_valid_xywh
from labeler.ui import GREEN, RED, WHITE, YELLOW, draw_bbox, draw_text
from labeler.video import get_frame_count, open_video, read_frame_at


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render annotations onto a video.")
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Annotation file path. Defaults to <video_name>.annotations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to render.",
    )

    return parser.parse_args(argv)


def video_writer_fourcc(codec: str) -> int:
    return int(getattr(cv2, "VideoWriter_fourcc")(*codec))


def annotation_for_frame(
    annotations: list[Annotation],
    frame_index: int,
) -> Annotation | None:
    if frame_index >= len(annotations):
        return None

    return annotations[frame_index]


def draw_annotation(
    frame,
    annotation: Annotation | None,
) -> None:
    if annotation is None:
        draw_text(frame, "UNLABELED", (10, 60), YELLOW)
        return

    if annotation.status == "V":
        x, y, width, height = annotation_bbox_to_xywh(
            annotation.x_center,
            annotation.y_center,
            annotation.width,
            annotation.height,
        )

        frame_height, frame_width = frame.shape[:2]
        bbox = clamp_xywh_to_frame(
            x=x,
            y=y,
            width=width,
            height=height,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if is_valid_xywh(*bbox):
            draw_bbox(frame, bbox, GREEN, thickness=3)
            draw_text(frame, "VISIBLE", (10, 60), GREEN)
        else:
            draw_text(frame, "VISIBLE - INVALID BBOX", (10, 60), RED)

        return

    if annotation.status == "S":
        draw_text(frame, "SKIPPED", (10, 60), YELLOW)
        return

    if annotation.status == "I":
        draw_text(frame, "INVISIBLE", (10, 60), RED)
        return

    raise ValueError(f"Unexpected annotation status: {annotation.status}")


def render_video(
    video_path: Path,
    annotation_path: Path,
    output_path: Path,
    max_frames: int | None,
) -> None:
    annotations = read_annotations(annotation_path)
    capture = open_video(video_path)

    try:
        frame_count = get_frame_count(capture)
        first_frame = read_frame_at(capture, 0)

        if first_frame is None:
            raise ValueError(f"Could not read first frame from {video_path}")

        height, width = first_frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            video_writer_fourcc("mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")

        try:
            frames_to_render = frame_count
            if max_frames is not None:
                frames_to_render = min(frames_to_render, max_frames)

            for frame_index in range(frames_to_render):
                frame = read_frame_at(capture, frame_index)
                if frame is None:
                    break

                draw_text(frame, f"Frame {frame_index + 1}", (10, 30), WHITE)
                draw_annotation(frame, annotation_for_frame(annotations, frame_index))

                writer.write(frame)

        finally:
            writer.release()

    finally:
        capture.release()


def main() -> None:
    args = parse_args()

    video_path: Path = args.video
    annotation_path: Path = args.annotations or default_annotation_path(video_path)

    render_video(
        video_path=video_path,
        annotation_path=annotation_path,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
