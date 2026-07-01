from pathlib import Path
from typing import Any

import cv2
import numpy as np


Frame = np.ndarray


def open_video(video_path: Path) -> Any:
    """Open a video file with OpenCV.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if OpenCV cannot open the video.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video file: {video_path}")

    return capture


def get_frame_count(capture: Any) -> int:
    """Return the number of frames reported by OpenCV."""
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_frame_size(capture: Any) -> tuple[int, int]:
    """Return frame size as width, height."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


def read_frame_at(capture: Any, frame_index: int) -> Frame | None:
    """Read a frame by zero-based frame index.

    Returns None if the frame cannot be read.
    """
    if frame_index < 0:
        raise ValueError(f"Frame index must be non-negative, got {frame_index}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()

    if not ok:
        return None

    return frame
