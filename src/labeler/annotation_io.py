from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Annotation:
    """One frame-level annotation.

    Status values:
    - V: object is visible, bbox fields contain center_x, center_y, width, height.
    - S: frame was skipped, bbox fields must be -1.
    - I: object is invisible/occluded/out of frame, bbox fields must be -1.
    """

    status: str
    x_center: int = -1
    y_center: int = -1
    width: int = -1
    height: int = -1

    def __post_init__(self) -> None:
        """Validate annotation consistency after dataclass initialization."""
        if self.status not in {"V", "S", "I"}:
            raise ValueError(f"Invalid annotation status: {self.status}")

        if self.status in {"S", "I"}:
            placeholders = (self.x_center, self.y_center, self.width, self.height)
            if placeholders != (-1, -1, -1, -1):
                raise ValueError(f"{self.status} annotations must use -1 placeholders")

        if self.status == "V":
            if self.width <= 0 or self.height <= 0:
                raise ValueError("Visible annotations must have positive width and height")
            if self.x_center < 0 or self.y_center < 0:
                raise ValueError("Visible annotations must have non-negative center coordinates")


def visible_annotation(x_center: int, y_center: int, width: int, height: int) -> Annotation:
    """Create a visible-object annotation."""
    return Annotation("V", x_center, y_center, width, height)


def skipped_annotation() -> Annotation:
    """Create a skipped-frame annotation."""
    return Annotation("S", -1, -1, -1, -1)


def invisible_annotation() -> Annotation:
    """Create an invisible-object annotation."""
    return Annotation("I", -1, -1, -1, -1)


def default_annotation_path(video_path: Path) -> Path:
    """Return the default annotation path for a video.

    Example:
        sample/video.mp4 -> sample/video.annotations
    """
    return video_path.with_suffix(".annotations")


def parse_annotation_line(line: str) -> Annotation:
    """Parse one annotation file line into an Annotation object.

    Expected format:
        V x_center y_center width height
        S -1 -1 -1 -1
        I -1 -1 -1 -1
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields in annotation line, got {len(parts)}: {line!r}")

    status = parts[0]
    if status not in {"V", "S", "I"}:
        raise ValueError(f"Invalid annotation status: {status!r}")

    try:
        x_center, y_center, width, height = [int(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"Annotation coordinates must be integers: {line!r}") from exc

    return Annotation(status, x_center, y_center, width, height)


def format_annotation(annotation: Annotation) -> str:
    """Convert an Annotation object to one annotation file line."""
    return (
        f"{annotation.status} "
        f"{annotation.x_center} "
        f"{annotation.y_center} "
        f"{annotation.width} "
        f"{annotation.height}"
    )


def read_annotations(path: Path) -> list[Annotation]:
    """Read an annotation file.

    Blank lines are ignored. Invalid lines include the line number in the raised error.
    """
    annotations: list[Annotation] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                annotations.append(parse_annotation_line(line))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid annotation file {path} at line {line_number}: {exc}"
                ) from exc

    return annotations


def write_annotations(path: Path, annotations: list[Annotation]) -> None:
    """Write annotations to disk, one line per frame."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [format_annotation(annotation) for annotation in annotations]
    text = "\n".join(lines)

    if text:
        text += "\n"

    path.write_text(text, encoding="utf-8")
