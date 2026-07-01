from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np

from labeler.geometry import XYWH, is_valid_xywh


Frame = np.ndarray
TrackerFactory = Callable[[], Any]


def create_csrt_tracker() -> Any:
    """Create an OpenCV CSRT tracker across OpenCV API variants."""
    create_tracker = getattr(cv2, "TrackerCSRT_create", None)
    if create_tracker is not None:
        return create_tracker()

    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        create_legacy_tracker = getattr(legacy, "TrackerCSRT_create", None)
        if create_legacy_tracker is not None:
            return create_legacy_tracker()

    raise RuntimeError(
        "OpenCV CSRT tracker is unavailable. Install opencv-contrib-python, not opencv-python."
    )


def normalize_tracker_bbox(bbox: Sequence[float]) -> XYWH:
    """Convert an OpenCV tracker bbox to integer xywh format."""
    if len(bbox) != 4:
        raise ValueError(f"Expected tracker bbox with 4 values, got {len(bbox)}")

    x, y, width, height = bbox

    return (
        int(round(x)),
        int(round(y)),
        int(round(width)),
        int(round(height)),
    )


class OpenCVObjectTracker:
    """Small wrapper around an OpenCV single-object tracker."""

    def __init__(self, tracker_factory: TrackerFactory = create_csrt_tracker) -> None:
        self._tracker_factory = tracker_factory
        self._tracker: Any | None = None
        self._last_bbox: XYWH | None = None

    def has_tracker(self) -> bool:
        """Return True when the tracker is initialized."""
        return self._tracker is not None

    def last_bbox(self) -> XYWH | None:
        """Return the most recent known bbox."""
        return self._last_bbox

    def reset(self) -> None:
        """Clear tracker state."""
        self._tracker = None
        self._last_bbox = None

    def initialize(self, frame: Frame, bbox: XYWH) -> None:
        """Initialize tracker from a known bbox."""
        if not is_valid_xywh(*bbox):
            raise ValueError(f"Cannot initialize tracker with invalid bbox: {bbox}")

        tracker = self._tracker_factory()
        result = tracker.init(frame, bbox)

        if result is False:
            self.reset()
            raise RuntimeError("OpenCV tracker failed to initialize")

        self._tracker = tracker
        self._last_bbox = bbox

    def update(self, frame: Frame) -> XYWH | None:
        """Predict bbox on the next frame.

        Returns None if the tracker is not initialized or tracking fails.
        """
        if self._tracker is None:
            return None

        ok, raw_bbox = self._tracker.update(frame)

        if not ok:
            self.reset()
            return None

        bbox = normalize_tracker_bbox(raw_bbox)

        if not is_valid_xywh(*bbox):
            self.reset()
            return None

        self._last_bbox = bbox
        return bbox
