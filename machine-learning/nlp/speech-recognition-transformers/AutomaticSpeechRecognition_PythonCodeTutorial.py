# %%
# !pip install transformers==4.11.2 datasets soundfile sentencepiece torchaudio pyaudio

# %%
from transformers import *
import torch
import soundfile as sf
# import librosa
import os
import torchaudio

# %%
# model_name = "facebook/wav2vec2-base-960h" # 360MB
model_name = "facebook/wav2vec2-large-960h-lv60-self" # 1.18GB

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# %%
# audio_url = "http://www.fit.vutbr.cz/~motlicek/sympatex/f2bjrop1.0.wav"
# audio_url = "http://www.fit.vutbr.cz/~motlicek/sympatex/f2bjrop1.1.wav"
# audio_url = "http://www.fit.vutbr.cz/~motlicek/sympatex/f2btrop6.0.wav"
# audio_url = "https://github.com/x4nth055/pythoncode-tutorials/raw/master/machine-learning/speech-recognition/16-122828-0002.wav"
audio_url = "https://github.com/x4nth055/pythoncode-tutorials/raw/master/machine-learning/speech-recognition/30-4447-0004.wav"
# audio_url = "https://github.com/x4nth055/pythoncode-tutorials/raw/master/machine-learning/speech-recognition/7601-291468-0006.wav"
# audio_url = "https://file-examples-com.github.io/uploads/2017/11/file_example_WAV_1MG.wav"
# audio_url = "http://www0.cs.ucl.ac.uk/teaching/GZ05/samples/lathe.wav"

# %%
# load our wav file
speech, sr = torchaudio.load(audio_url)
speech = speech.squeeze()
# or using librosa
# speech, sr = librosa.load(audio_file, sr=16000)
sr, speech.shape

# %%
# resample from whatever the audio sampling rate to 16000
resampler = torchaudio.transforms.Resample(sr, 16000)
speech = resampler(speech)
speech.shape

# %%
# tokenize our wav
input_values = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
input_values.shape

# %%
# perform inference
logits = model(input_values)["logits"]
logits.shape

# %%
# use argmax to get the predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)
predicted_ids.shape

# %%
# decode the IDs to text
transcription = processor.decode(predicted_ids[0])
transcription.lower()

# %%
def get_transcription(audio_path):
  # load our wav file
  speech, sr = torchaudio.load(audio_path)
  speech = speech.squeeze()
  # or using librosa
  # speech, sr = librosa.load(audio_file, sr=16000)
  # resample from whatever the audio sampling rate to 16000
  resampler = torchaudio.transforms.Resample(sr, 16000)
  speech = resampler(speech)
  # tokenize our wav
  input_values = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
  # perform inference
  logits = model(input_values)["logits"]
  # use argmax to get the predicted IDs
  predicted_ids = torch.argmax(logits, dim=-1)
  # decode the IDs to text
  transcription = processor.decode(predicted_ids[0])
  return transcription.lower()

# %%
get_transcription(audio_url)

# %%
import pyaudio
import wave

# the file name output you want to record into
filename = "recorded.wav"
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 16000
record_seconds = 10
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
for i in range(int(sample_rate / chunk * record_seconds)):
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

# %%
get_transcription("recorded.wav")

# %%



