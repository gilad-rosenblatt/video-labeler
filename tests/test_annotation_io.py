from pathlib import Path

import pytest

from skyways_cv_labeler.annotation_io import (
    Annotation,
    default_annotation_path,
    format_annotation,
    invisible_annotation,
    parse_annotation_line,
    read_annotations,
    skipped_annotation,
    visible_annotation,
    write_annotations,
)


def test_parse_visible_annotation() -> None:
    annotation = parse_annotation_line("V 250 150 50 75")

    assert annotation == Annotation("V", 250, 150, 50, 75)


def test_parse_skipped_annotation() -> None:
    annotation = parse_annotation_line("S -1 -1 -1 -1")

    assert annotation == Annotation("S", -1, -1, -1, -1)


def test_parse_invisible_annotation() -> None:
    annotation = parse_annotation_line("I -1 -1 -1 -1")

    assert annotation == Annotation("I", -1, -1, -1, -1)


def test_format_visible_annotation() -> None:
    annotation = visible_annotation(250, 150, 50, 75)

    assert format_annotation(annotation) == "V 250 150 50 75"


def test_format_skipped_annotation() -> None:
    annotation = skipped_annotation()

    assert format_annotation(annotation) == "S -1 -1 -1 -1"


def test_format_invisible_annotation() -> None:
    annotation = invisible_annotation()

    assert format_annotation(annotation) == "I -1 -1 -1 -1"


def test_reject_invalid_status() -> None:
    with pytest.raises(ValueError, match="Invalid annotation status"):
        parse_annotation_line("X -1 -1 -1 -1")


def test_reject_wrong_number_of_fields() -> None:
    with pytest.raises(ValueError, match="Expected 5 fields"):
        parse_annotation_line("V 1 2 3")


def test_reject_non_integer_values() -> None:
    with pytest.raises(ValueError, match="must be integers"):
        parse_annotation_line("V 1 2 nope 4")


def test_reject_visible_annotation_with_zero_width() -> None:
    with pytest.raises(ValueError, match="positive width and height"):
        parse_annotation_line("V 10 20 0 5")


def test_reject_visible_annotation_with_negative_center() -> None:
    with pytest.raises(ValueError, match="non-negative center coordinates"):
        parse_annotation_line("V -1 20 10 10")


def test_reject_skipped_annotation_without_placeholders() -> None:
    with pytest.raises(ValueError, match="must use -1 placeholders"):
        Annotation("S", 10, 20, 30, 40)


def test_reject_invisible_annotation_without_placeholders() -> None:
    with pytest.raises(ValueError, match="must use -1 placeholders"):
        Annotation("I", 10, 20, 30, 40)


def test_default_annotation_path() -> None:
    assert default_annotation_path(Path("sample/video.mp4")) == Path("sample/video.annotations")


def test_read_write_annotations_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "test.annotations"
    annotations = [
        visible_annotation(250, 150, 50, 75),
        skipped_annotation(),
        invisible_annotation(),
    ]

    write_annotations(path, annotations)
    loaded = read_annotations(path)

    assert loaded == annotations


def test_read_annotations_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "test.annotations"
    path.write_text("V 10 20 30 40\n\nS -1 -1 -1 -1\n", encoding="utf-8")

    loaded = read_annotations(path)

    assert loaded == [
        visible_annotation(10, 20, 30, 40),
        skipped_annotation(),
    ]


def test_read_annotations_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.annotations"
    path.write_text("V 10 20 30 40\nX -1 -1 -1 -1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_annotations(path)
