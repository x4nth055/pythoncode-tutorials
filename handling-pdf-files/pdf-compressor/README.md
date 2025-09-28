# [How to Compress PDF Files in Python](https://www.thepythoncode.com/article/compress-pdf-files-in-python)

This directory contains two approaches:

- Legacy (commercial): `pdf_compressor.py` uses PDFTron/PDFNet. PDFNet now requires a license key and the old pip package is not freely available, so this may not work without a license.
- Recommended (open source): `pdf_compressor_ghostscript.py` uses Ghostscript to compress PDFs.

## Ghostscript method (recommended)

Prerequisite: Install Ghostscript

- macOS (Homebrew):
  - `brew install ghostscript`
- Ubuntu/Debian:
  - `sudo apt-get update && sudo apt-get install -y ghostscript`
- Windows:
  - Download and install from https://ghostscript.com/releases/
  - Ensure `gswin64c.exe` (or `gswin32c.exe`) is in your PATH.

No Python packages are required for this method, only Ghostscript.

### Usage

To compress `bert-paper.pdf` into `bert-paper-min.pdf` with default quality (`power=2`):

```
python pdf_compressor_ghostscript.py bert-paper.pdf bert-paper-min.pdf
```

Optional quality level `[power]` controls compression/quality tradeoff (maps to Ghostscript `-dPDFSETTINGS`):

- 0 = `/screen` (smallest, lowest quality)
- 1 = `/ebook` (good quality)
- 2 = `/printer` (high quality) [default]
- 3 = `/prepress` (very high quality)
- 4 = `/default` (Ghostscript default)

Example:

```
python pdf_compressor_ghostscript.py bert-paper.pdf bert-paper-min.pdf 1
```

In testing, `bert-paper.pdf` (~757 KB) compressed to ~407 KB with `power=1`.

## Legacy PDFNet method (requires license)

If you have a valid license and the PDFNet SDK installed, you can use the original `pdf_compressor.py` script. Note that the previously referenced `PDFNetPython3` pip package is not freely available and may not install via pip. Refer to the vendor's documentation for installation and licensing.