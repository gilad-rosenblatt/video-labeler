from pathlib import Path

import cv2
import numpy as np
import pytest

from labeler.video import (
    get_frame_count,
    get_frame_size,
    open_video,
    read_frame_at,
)


def create_test_video(path: Path, frame_count: int = 5) -> None:
    width = 64
    height = 48
    fps = 10.0

    writer = cv2.VideoWriter(
        str(path),
        getattr(cv2, "VideoWriter_fourcc")(*"mp4v"),
        fps,
        (width, height),
    )

    assert writer.isOpened()

    for frame_index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = frame_index * 20
        writer.write(frame)

    writer.release()


def test_open_video_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        open_video(tmp_path / "missing.mp4")


def test_open_video_existing_video(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path)

    capture = open_video(video_path)

    try:
        assert capture.isOpened()
    finally:
        capture.release()


def test_get_frame_count(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path, frame_count=5)

    capture = open_video(video_path)

    try:
        assert get_frame_count(capture) == 5
    finally:
        capture.release()


def test_get_frame_size(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path)

    capture = open_video(video_path)

    try:
        assert get_frame_size(capture) == (64, 48)
    finally:
        capture.release()


def test_read_frame_at(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path, frame_count=5)

    capture = open_video(video_path)

    try:
        frame = read_frame_at(capture, 2)

        assert frame is not None
        assert frame.shape == (48, 64, 3)
    finally:
        capture.release()


def test_read_frame_at_negative_index_raises(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path)

    capture = open_video(video_path)

    try:
        with pytest.raises(ValueError, match="Frame index must be non-negative"):
            read_frame_at(capture, -1)
    finally:
        capture.release()


def test_read_frame_at_out_of_range_returns_none(tmp_path: Path) -> None:
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path, frame_count=5)

    capture = open_video(video_path)

    try:
        frame = read_frame_at(capture, 100)

        assert frame is None
    finally:
        capture.release()
