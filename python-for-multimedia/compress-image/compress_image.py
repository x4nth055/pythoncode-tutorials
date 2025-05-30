import os
from PIL import Image
import argparse
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_size_format(b, factor=1024, suffix="B"):
    """Scale bytes to its proper byte format."""
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"

def compress_image(
    input_path,
    output_dir=None,
    quality=85,
    resize_ratio=1.0,
    width=None,
    height=None,
    to_jpg=False,
    preserve_metadata=True,
    lossless=False,
):
    """Compress an image with advanced options."""
    try:
        img = Image.open(input_path)
        logger.info(f"[*] Processing: {os.path.basename(input_path)}")
        logger.info(f"[*] Original size: {get_size_format(os.path.getsize(input_path))}")

        # Resize if needed
        if resize_ratio < 1.0:
            new_size = (int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
            img = img.resize(new_size, Image.LANCZOS)
            logger.info(f"[+] Resized to: {new_size}")
        elif width and height:
            img = img.resize((width, height), Image.LANCZOS)
            logger.info(f"[+] Resized to: {width}x{height}")

        # Prepare output path
        filename, ext = os.path.splitext(os.path.basename(input_path))
        output_ext = ".jpg" if to_jpg else ext
        output_filename = f"{filename}_compressed{output_ext}"
        output_path = os.path.join(output_dir or os.path.dirname(input_path), output_filename)

        # Save with options
        save_kwargs = {"quality": quality, "optimize": True}
        if not preserve_metadata:
            save_kwargs["exif"] = b""  # Strip metadata
        if lossless and ext.lower() in (".png", ".webp"):
            save_kwargs["lossless"] = True

        try:
            img.save(output_path, **save_kwargs)
        except OSError:
            img = img.convert("RGB")
            img.save(output_path, **save_kwargs)

        logger.info(f"[+] Saved to: {output_path}")
        logger.info(f"[+] New size: {get_size_format(os.path.getsize(output_path))}")
    except Exception as e:
        logger.error(f"[!] Error processing {input_path}: {e}")

def batch_compress(
    input_paths,
    output_dir=None,
    quality=85,
    resize_ratio=1.0,
    width=None,
    height=None,
    to_jpg=False,
    preserve_metadata=True,
    lossless=False,
):
    """Compress multiple images."""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for path in tqdm(input_paths, desc="Compressing images"):
        compress_image(path, output_dir, quality, resize_ratio, width, height, to_jpg, preserve_metadata, lossless)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Image Compressor with Batch Processing")
    parser.add_argument("input", nargs='+', help="Input image(s) or directory")
    parser.add_argument("-o", "--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("-q", "--quality", type=int, default=85, help="Compression quality (0-100)")
    parser.add_argument("-r", "--resize-ratio", type=float, default=1.0, help="Resize ratio (0-1)")
    parser.add_argument("-w", "--width", type=int, help="Output width (requires --height)")
    parser.add_argument("-hh", "--height", type=int, help="Output height (requires --width)")
    parser.add_argument("-j", "--to-jpg", action="store_true", help="Convert output to JPEG")
    parser.add_argument("-m", "--no-metadata", action="store_false", help="Strip metadata")
    parser.add_argument("-l", "--lossless", action="store_true", help="Use lossless compression (PNG/WEBP)")

    args = parser.parse_args()
    input_paths = []
    for path in args.input:
        if os.path.isdir(path): input_paths.extend(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".jpg",".jpeg",".png",".webp")))
        else: input_paths.append(path)
    if not input_paths: logger.error("No valid images found!"); exit(1)
    batch_compress(input_paths, args.output_dir, args.quality, args.resize_ratio, args.width, args.height, args.to_jpg, args.no_metadata, args.lossless)
