import pyaudio
import wave
import argparse

parser = argparse.ArgumentParser(description="an Audio Recorder using Python")
parser.add_argument("-o", "--output", help="Output file (with .wav)", default="recorded.wav")
parser.add_argument("-d", "--duration", help="Duration to record in seconds (can be float)", default=5)

args = parser.parse_args()
# the file name output you want to record into
filename = args.output
# number of seconds to record
record_seconds = float(args.duration)

# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 44100

# initialize PyAudio object
p = pyaudio.PyAudio()

# open stream object as input & output
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)

frames = []
print("Recording...")
for i in range(int(44100 / chunk * record_seconds)):
    data = stream.read(chunk)
    # if you want to hear your voice while recording
    # stream.write(data)
    frames.append(data)
print("Finished recording.")
# stop and close stream
stream.stop_stream()
stream.close()
# terminate pyaudio object
p.terminate()

# save audio file
# open the file in 'write bytes' mode
wf = wave.open(filename, "wb")
# set the channels
wf.setnchannels(channels)
# set the sample format
wf.setsampwidth(p.get_sample_size(FORMAT))
# set the sample rate
wf.setframerate(sample_rate)
# write the frames as bytes
wf.writeframes(b"".join(frames))
# close the file
wf.close()