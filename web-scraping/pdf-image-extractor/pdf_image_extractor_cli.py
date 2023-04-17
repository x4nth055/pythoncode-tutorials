import os
import fitz  # PyMuPDF
import io
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Extract images from a PDF file.")
parser.add_argument("file", help="PDF file to extract images from.")
parser.add_argument("-o", "--output", help="Output directory for the extracted images.", default="extracted_images")
parser.add_argument("-f", "--format", help="Desired output image format, default is PNG.", default="png")
parser.add_argument("-w", "--width", help="Minimum width for extracted images, default is 100.", default=100, type=int)
parser.add_argument("-he", "--height", help="Minimum height for extracted images, default is 100.", default=100, type=int)
# Parse the arguments
args = parser.parse_args()

# Output directory for the extracted images
output_dir = args.output
# Desired output image format
output_format = args.format
# Minimum width and height for extracted images
min_width = args.width
min_height = args.height
# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# File path you want to extract images from
file = args.file
# Open the file
pdf_file = fitz.open(file)
# Iterate over PDF pages
for page_index in range(len(pdf_file)):
    # Get the page itself
    page = pdf_file[page_index]
    # Get image list
    image_list = page.get_images(full=True)
    # Print the number of images found on this page
    if image_list:
        print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
    else:
        print(f"[!] No images found on page {page_index}")
    # Iterate over the images on the page
    for image_index, img in enumerate(image_list, start=1):
        # Get the XREF of the image
        xref = img[0]
        # Extract the image bytes
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image["image"]
        # Get the image extension
        image_ext = base_image["ext"]
        # Load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        # Check if the image meets the minimum dimensions and save it
        if image.width >= min_width and image.height >= min_height:
            image.save(
                open(os.path.join(output_dir, f"image{page_index + 1}_{image_index}.{output_format}"), "wb"),
                format=output_format.upper())
        else:
            print(f"[-] Skipping image {image_index} on page {page_index} due to its small size.")
