from pydub import AudioSegment
from pydub.playback import play
import sys

# read MP3 file
song = AudioSegment.from_mp3(sys.argv[1])
# song = AudioSegment.from_wav("audio_file.wav")
# you can also read from other formats such as MP4
# song = AudioSegment.from_file("audio_file.mp4", "mp4")
play(song)
