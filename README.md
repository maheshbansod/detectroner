# Detectroner

Detectroner is a Python application that uses Detectron2 to detect objects in videos.

## Requirements

- Python 3.12
- uv

## Installation

```bash
sh setup.sh
```

## Usage

```bash
uv run main.py input-video.mp4
```
This will emit output files in the directory `output`.
The command will write the names of the objects that were detected to stdout.

You may use example videos in the `example_data` directory.

## Output

It emits output files in the directory `output`.
It emits
- JSON file with the detected objects, the timestamps when they were detected, and the image file name that contains the object.
- An image file for every detected object.