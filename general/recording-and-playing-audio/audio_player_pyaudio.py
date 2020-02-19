import pyaudio
import wave
import sys

filename = sys.argv[1]

# set the chunk size of 1024 samples
chunk = 1024

# open the audio file
wf = wave.open(filename, "rb")

# initialize PyAudio object
p = pyaudio.PyAudio()

# open stream object
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# read data in chunks
data = wf.readframes(chunk)

# writing to the stream (playing audio)
while data:
    stream.write(data)
    data = wf.readframes(chunk)

# close stream
stream.close()
p.terminate()
                