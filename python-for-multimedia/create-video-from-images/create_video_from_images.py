import cv2
import argparse
import glob
from pathlib import Path
import shutil

# Create an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser(description='Create a video from a set of images')

# Define the command-line arguments
parser.add_argument('output', type=str, help='Output path for video file')
parser.add_argument('input', nargs='+', type=str, help='Glob pattern for input images')
parser.add_argument('-fps', type=int, help='FPS for video file', default=24)

# Parse the command-line arguments
args = parser.parse_args()

# Create a list of all the input image files
FILES = []
for i in args.input:
    FILES += glob.glob(i)

# Get the filename from the output path
filename = Path(args.output).name
print(f'Creating video "{filename}" from images "{FILES}"')

# Load the first image to get the frame size
frame = cv2.imread(FILES[0])
height, width, layers = frame.shape

# Create a VideoWriter object to write the video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=args.fps, frameSize=(width, height))

# Loop through the input images and add them to the video
for image_path in FILES:
    print(f'Adding image "{image_path}" to video "{args.output}"... ')
    video.write(cv2.imread(image_path))

# Release the VideoWriter and move the output file to the specified location
cv2.destroyAllWindows()
video.release()
shutil.move(filename, args.output)
