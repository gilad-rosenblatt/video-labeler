from pathlib import Path

import cv2
from PIL import Image


def main() -> None:
    input_path = Path("sample/real_demo_annotated.mp4")
    output_path = Path("sample/real_demo_preview.gif")

    fps = 10
    max_seconds = 8
    output_width = 480

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {input_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(source_fps / fps)))
    max_frames = fps * max_seconds

    frames: list[Image.Image] = []
    frame_index = 0

    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % frame_step == 0:
            height, width = frame.shape[:2]
            scale = output_width / width
            output_height = int(round(height * scale))

            resized = cv2.resize(
                frame,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        frame_index += 1

    capture.release()

    if not frames:
        raise RuntimeError("No frames were extracted")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
