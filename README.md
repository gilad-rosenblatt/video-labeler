# Video Labeler

Interactive single-object video labeling and annotation validation tools.

The project provides two command-line tools:

- `label`: interactively label one object of interest across a video.
- `validate`: review saved annotations by overlaying them on the video.

## Demo

![Annotated demo preview](sample/real_demo_preview.gif)

[▶ Watch annotated demo video](sample/real_demo_annotated.mp4)

The demo video was generated from:

```text
sample/real_demo.mp4
sample/real_demo.annotations
```

## Setup

Requires Python 3.10+.

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Labeling a video

```bash
label path/to/video.mp4
```

By default, annotations are written next to the video:

```text
path/to/video.annotations
```

You can also specify the output path:

```bash
label path/to/video.mp4 --annotations path/to/output.annotations
```

## Labeling controls

When there is no tracker prediction:

```text
l / L   manually label visible object
s / S   skip frame
i / I   mark object invisible
q / Q   quit
Esc     quit
```

Manual labeling uses center-drag selection:

```text
click object center -> drag outward to box corner -> release
```

After a visible object has been labeled, the tracker predicts the object location on the next frame.

When a tracker prediction is available:

```text
a / A   accept predicted box
f / F   fix predicted box manually
s / S   skip frame
i / I   mark object invisible
q / Q   quit
Esc     quit
```

Skipped and invisible frames reset the tracker.

## Validating annotations

```bash
validate path/to/video.mp4
```

or:

```bash
validate path/to/video.mp4 --annotations path/to/output.annotations
```

Validation controls:

```text
n       next frame
N       jump forward 10 frames
p       previous frame
P       jump backward 10 frames
q       quit
Esc     quit
```

The validator overlays visible-object boxes and displays skipped, invisible, or unlabeled frame status.

## Annotation format

Each line corresponds to one video frame.

Visible object:

```text
V x_center y_center width height
```

Example:

```text
V 250 150 50 75
```

Skipped frame:

```text
S -1 -1 -1 -1
```

Invisible object:

```text
I -1 -1 -1 -1
```

## Sample and demo videos

A generated sample video and annotation file are included:

```text
sample/sample.mp4
sample/sample.annotations
```

Run:

```bash
validate sample/sample.mp4
```

To generate the sample video again:

```bash
python scripts/create_sample_video.py
```

A real demo video and annotation file can also be included:

```text
sample/real_demo.mp4
sample/real_demo.annotations
```

To label the real demo video:

```bash
label sample/real_demo.mp4 --annotations sample/real_demo.annotations
```

To validate the real demo annotations interactively:

```bash
validate sample/real_demo.mp4 --annotations sample/real_demo.annotations
```

To render an annotated demo video for the README:

```bash
python scripts/render_annotations.py sample/real_demo.mp4 --annotations sample/real_demo.annotations --output sample/real_demo_annotated.mp4
```

To generate an animated GIF preview from the annotated demo video:

```bash
python scripts/create_gif_preview.py
```

The rendered demo is linked near the top of this README:

```text
sample/real_demo_annotated.mp4
sample/real_demo_preview.gif
```

## Development

Run tests:

```bash
python -m pytest
```

Run linting:

```bash
python -m ruff check .
```

## Developer notes

Project structure:

```text
src/labeler/
  annotation_io.py      annotation parsing/writing
  bbox_selector.py      center-click + drag bbox selection UI
  geometry.py           bbox coordinate conversions and clamping
  label_tool.py         interactive labeling CLI
  tracker.py            OpenCV CSRT tracker wrapper
  ui.py                 shared OpenCV drawing helpers
  validate_tool.py      annotation validation CLI
  video.py              video loading and frame access helpers

tests/                  unit tests for core logic
scripts/                sample/demo generation utilities
sample/                 sample videos and annotations
```

Recommended development flow:

```bash
python -m pytest
python -m ruff check .
```

Manual smoke tests:

```bash
validate sample/sample.mp4
label sample/sample.mp4 --annotations sample/manual_test.annotations
validate sample/sample.mp4 --annotations sample/manual_test.annotations
```

The code keeps OpenCV GUI behavior thin and pushes reusable logic into small functions so it can be tested without opening windows.

## Design choices

The labeler uses a single-object OpenCV CSRT tracker after the first manual label. The tracker is used only as an annotation aid. The user still accepts or fixes predictions frame by frame.

The manual bounding-box interaction uses a center-click and drag-out workflow: the user clicks the object center and drags outward to define the box dimensions.

The annotation parser and writer are shared between the labeler and validator so both tools use the same file-format rules.

Skipped and invisible frames reset tracker state because object identity is uncertain after a skipped frame or absence.

## Known limitations

- The tool is intended for one object of interest, not multi-object annotation.
- Tracker predictions can drift and should be reviewed by the user.
- The validator is visual/manual; it does not compute annotation accuracy metrics.