# Compress Image

Advanced Image Compressor with Batch Processing

This script provides advanced image compression and resizing features using Python and Pillow.

## Features

- Batch processing of multiple images or directories
- Lossy and lossless compression (PNG/WebP)
- Optional JPEG conversion
- Resize by ratio or explicit dimensions
- Preserve or strip metadata (EXIF)
- Custom output directory
- Progress bar using `tqdm`
- Detailed logging

## Requirements

- Python 3.6+
- [Pillow](https://pypi.org/project/Pillow/)
- [tqdm](https://pypi.org/project/tqdm/)

Install dependencies:

```bash
pip install pillow tqdm
```

## Usage

```bash
python compress_image.py [options] <input> [<input> ...]
```

## Options
- `-o`, `--output-dir`: Output directory (default: same as input)
- `-q`, `--quality`: Compression quality (0-100, default: 85)
- `-r`, `--resize-ratio`: Resize ratio (0-1, default: 1.0)
- `-w`, `--width`: Output width (requires `--height`)
- `-hh`, `--height`: Output height (requires `--width`)
- `-j`, `--to-jpg`: Convert output to JPEG
- `-m`, `--no-metadata`: Strip metadata (default: preserve)
- `-l`, `--lossless`: Use lossless compression (PNG/WEBP)

## Examples

```bash
python compress_image.py image.jpg -r 0.5 -q 80 -j
python compress_image.py images/ -o output/ -m
python compress_image.py image.png -l
```

## License

MIT License.
