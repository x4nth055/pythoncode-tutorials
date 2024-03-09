# Import necessary libraries.
import argparse
from PIL import Image


# Function to clear Metadata from a specified image.
def clear_all_metadata(imgname):
  
    # Open the image file
    img = Image.open(imgname)
    
    # Read the image data, excluding metadata.
    data = list(img.getdata())
    
    # Create a new image with the same mode and size but without metadata.
    img_without_metadata = Image.new(img.mode, img.size)
    img_without_metadata.putdata(data)
    
    # Save the new image over the original file, effectively removing metadata.
    img_without_metadata.save(imgname)
    
    print(f"Metadata successfully cleared from '{imgname}'.")

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Remove metadata from an image file.")
parser.add_argument("img", help="Image file from which to remove metadata")

# Parse arguments
args = parser.parse_args()

# If an image file is provided, clear its metadata
if args.img:
    clear_all_metadata(args.img)
