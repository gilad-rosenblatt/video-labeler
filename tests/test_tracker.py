import numpy as np
import pytest

from labeler.tracker import OpenCVObjectTracker, normalize_tracker_bbox


class FakeTracker:
    def __init__(
        self,
        init_result: bool | None = True,
        update_result: tuple[bool, tuple[float, float, float, float]] = (
            True,
            (10.0, 20.0, 30.0, 40.0),
        ),
    ) -> None:
        self.init_result = init_result
        self.update_result = update_result
        self.initialized_frame: np.ndarray | None = None
        self.initialized_bbox: tuple[int, int, int, int] | None = None

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool | None:
        self.initialized_frame = frame
        self.initialized_bbox = bbox
        return self.init_result

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        del frame
        return self.update_result


def test_normalize_tracker_bbox() -> None:
    assert normalize_tracker_bbox((10.2, 20.6, 30.4, 40.5)) == (10, 21, 30, 40)


def test_normalize_tracker_bbox_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="Expected tracker bbox with 4 values"):
        normalize_tracker_bbox((1.0, 2.0, 3.0))


def test_tracker_starts_empty() -> None:
    tracker = OpenCVObjectTracker(tracker_factory=FakeTracker)

    assert not tracker.has_tracker()
    assert tracker.last_bbox() is None


def test_initialize_tracker() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    fake_tracker = FakeTracker()

    tracker = OpenCVObjectTracker(tracker_factory=lambda: fake_tracker)
    tracker.initialize(frame, (10, 20, 30, 40))

    assert tracker.has_tracker()
    assert tracker.last_bbox() == (10, 20, 30, 40)
    assert fake_tracker.initialized_frame is frame
    assert fake_tracker.initialized_bbox == (10, 20, 30, 40)


def test_initialize_tracker_accepts_none_result_from_opencv() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    tracker = OpenCVObjectTracker(tracker_factory=lambda: FakeTracker(init_result=None))
    tracker.initialize(frame, (10, 20, 30, 40))

    assert tracker.has_tracker()
    assert tracker.last_bbox() == (10, 20, 30, 40)


def test_initialize_tracker_rejects_invalid_bbox() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    tracker = OpenCVObjectTracker(tracker_factory=FakeTracker)

    with pytest.raises(ValueError, match="invalid bbox"):
        tracker.initialize(frame, (10, 20, 0, 40))


def test_initialize_tracker_raises_when_opencv_init_fails() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    tracker = OpenCVObjectTracker(tracker_factory=lambda: FakeTracker(init_result=False))

    with pytest.raises(RuntimeError, match="failed to initialize"):
        tracker.initialize(frame, (10, 20, 30, 40))

    assert not tracker.has_tracker()
    assert tracker.last_bbox() is None


def test_update_without_initialized_tracker_returns_none() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    tracker = OpenCVObjectTracker(tracker_factory=FakeTracker)

    assert tracker.update(frame) is None


def test_update_returns_predicted_bbox() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    fake_tracker = FakeTracker(update_result=(True, (11.2, 21.6, 31.4, 41.5)))
    tracker = OpenCVObjectTracker(tracker_factory=lambda: fake_tracker)

    tracker.initialize(frame, (10, 20, 30, 40))
    prediction = tracker.update(frame)

    assert prediction == (11, 22, 31, 42)
    assert tracker.last_bbox() == (11, 22, 31, 42)


def test_update_resets_when_tracking_fails() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    fake_tracker = FakeTracker(update_result=(False, (0.0, 0.0, 0.0, 0.0)))
    tracker = OpenCVObjectTracker(tracker_factory=lambda: fake_tracker)

    tracker.initialize(frame, (10, 20, 30, 40))

    assert tracker.update(frame) is None
    assert not tracker.has_tracker()
    assert tracker.last_bbox() is None


def test_update_resets_when_prediction_is_invalid() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    fake_tracker = FakeTracker(update_result=(True, (10.0, 20.0, 0.0, 40.0)))
    tracker = OpenCVObjectTracker(tracker_factory=lambda: fake_tracker)

    tracker.initialize(frame, (10, 20, 30, 40))

    assert tracker.update(frame) is None
    assert not tracker.has_tracker()
    assert tracker.last_bbox() is None


def test_reset_clears_tracker() -> None:
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    tracker = OpenCVObjectTracker(tracker_factory=FakeTracker)
    tracker.initialize(frame, (10, 20, 30, 40))

    tracker.reset()

    assert not tracker.has_tracker()
    assert tracker.last_bbox() is None
