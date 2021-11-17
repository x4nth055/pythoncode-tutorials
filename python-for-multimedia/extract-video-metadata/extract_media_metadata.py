import ffmpeg
import sys
from pprint import pprint # for printing Python dictionaries in a human-readable way

# read the audio/video file from the command line arguments
media_file = sys.argv[1]
# uses ffprobe command to extract all possible metadata from the media file
pprint(ffmpeg.probe(media_file)["streams"])
