import os
import sys
import subprocess
import shutil


def get_size_format(b, factor=1024, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"


def find_ghostscript_executable():
    candidates = [
        shutil.which('gs'),
        shutil.which('gswin64c'),
        shutil.which('gswin32c'),
    ]
    for c in candidates:
        if c:
            return c
    return None


def compress_file(input_file: str, output_file: str, power: int = 2):
    """Compress PDF using Ghostscript.

    power:
        0 -> /screen (lowest quality, highest compression)
        1 -> /ebook (good quality)
        2 -> /printer (high quality) [default]
        3 -> /prepress (very high quality)
        4 -> /default (Ghostscript default)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not output_file:
        output_file = input_file

    initial_size = os.path.getsize(input_file)

    gs = find_ghostscript_executable()
    if not gs:
        raise RuntimeError(
            "Ghostscript not found. Install it and ensure 'gs' (Linux/macOS) "
            "or 'gswin64c'/'gswin32c' (Windows) is in PATH."
        )

    settings_map = {
        0: '/screen',
        1: '/ebook',
        2: '/printer',
        3: '/prepress',
        4: '/default',
    }
    pdfsettings = settings_map.get(power, '/printer')

    cmd = [
        gs,
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        f'-dPDFSETTINGS={pdfsettings}',
        '-dNOPAUSE',
        '-dBATCH',
        '-dQUIET',
        f'-sOutputFile={output_file}',
        input_file,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ghostscript failed: {e}")
        return False

    compressed_size = os.path.getsize(output_file)
    ratio = 1 - (compressed_size / initial_size)
    summary = {
        "Input File": input_file,
        "Initial Size": get_size_format(initial_size),
        "Output File": output_file,
        "Compressed Size": get_size_format(compressed_size),
        "Compression Ratio": f"{ratio:.3%}",
    }

    print("## Summary ########################################################")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("###################################################################")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python pdf_compressor_ghostscript.py <input.pdf> <output.pdf> [power 0-4]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    power = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ok = compress_file(input_file, output_file, power)
    sys.exit(0 if ok else 2)