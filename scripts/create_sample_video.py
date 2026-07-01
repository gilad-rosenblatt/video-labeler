from pathlib import Path

import cv2
import numpy as np


def video_writer_fourcc(codec: str) -> int:
    return int(getattr(cv2, "VideoWriter_fourcc")(*codec))


def main() -> None:
    sample_dir = Path("sample")
    sample_dir.mkdir(exist_ok=True)

    video_path = sample_dir / "sample.mp4"
    annotation_path = sample_dir / "sample.annotations"

    width = 320
    height = 240
    frame_count = 60
    fps = 15.0

    box_width = 40
    box_height = 30

    writer = cv2.VideoWriter(
        str(video_path),
        video_writer_fourcc("mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    annotation_lines: list[str] = []

    for frame_index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        x_center = 40 + frame_index * 4
        y_center = 120

        x = x_center - box_width // 2
        y = y_center - box_height // 2

        cv2.rectangle(
            frame,
            (x, y),
            (x + box_width, y + box_height),
            (0, 0, 255),
            -1,
        )

        writer.write(frame)
        annotation_lines.append(f"V {x_center} {y_center} {box_width} {box_height}")

    writer.release()

    annotation_path.write_text("\n".join(annotation_lines) + "\n", encoding="utf-8")

    print(f"Wrote {video_path}")
    print(f"Wrote {annotation_path}")


if __name__ == "__main__":
    main()
