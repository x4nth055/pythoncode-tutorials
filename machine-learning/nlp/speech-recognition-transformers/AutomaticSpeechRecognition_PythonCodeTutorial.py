# %%
!pip install transformers==4.28.1 soundfile sentencepiece torchaudio pydub

# %%
from transformers import *
import torch
import soundfile as sf
# import librosa
import os
import torchaudio

# %% [markdown]
# # Wav2Vec2.0 Models
# 

# %%
# wav2vec2_model_name = "facebook/wav2vec2-base-960h" # 360MB
wav2vec2_model_name = "facebook/wav2vec2-large-960h-lv60-self" # pretrained 1.26GB
# wav2vec2_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english" # English-only, 1.26GB
# wav2vec2_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic" # Arabic-only, 1.26GB
# wav2vec2_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish" # Spanish-only, 1.26GB
# load model and tokenizer
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_name)

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
input_values = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
input_values.shape

# %%
# perform inference
logits = wav2vec2_model(input_values)["logits"]
logits.shape

# %%
# use argmax to get the predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)
predicted_ids.shape

# %%
# decode the IDs to text
transcription = wav2vec2_processor.decode(predicted_ids[0])
transcription.lower()

# %%
def load_audio(audio_path):
  """Load the audio file & convert to 16,000 sampling rate"""
  # load our wav file
  speech, sr = torchaudio.load(audio_path)
  resampler = torchaudio.transforms.Resample(sr, 16000)
  speech = resampler(speech)
  return speech.squeeze()

# %%
def get_transcription_wav2vec2(audio_path, model, processor):
  # load our wav file
  speech = load_audio(audio_path)
  # get the input features
  input_features = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
  # perform inference
  logits = model(input_features)["logits"]
  # use argmax to get the predicted IDs
  predicted_ids = torch.argmax(logits, dim=-1)
  # decode the IDs to text
  transcription = processor.batch_decode(predicted_ids)[0]
  return transcription.lower()

# %%
get_transcription_wav2vec2("http://www0.cs.ucl.ac.uk/teaching/GZ05/samples/lathe.wav", 
                           wav2vec2_model, 
                           wav2vec2_processor)

# %% [markdown]
# # Whisper Models

# %%
# whisper_model_name = "openai/whisper-tiny.en" # English-only, ~ 151 MB
# whisper_model_name = "openai/whisper-base.en" # English-only, ~ 290 MB
# whisper_model_name = "openai/whisper-small.en" # English-only, ~ 967 MB
# whisper_model_name = "openai/whisper-medium.en" # English-only, ~ 3.06 GB
# whisper_model_name = "openai/whisper-tiny" # multilingual, ~ 151 MB
# whisper_model_name = "openai/whisper-base" # multilingual, ~ 290 MB
# whisper_model_name = "openai/whisper-small" # multilingual, ~ 967 MB
whisper_model_name = "openai/whisper-medium" # multilingual, ~ 3.06 GB
# whisper_model_name = "openai/whisper-large-v2" # multilingual, ~ 6.17 GB
# load the whisper model and tokenizer
whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

# %%
# get the input features
input_features = whisper_processor(load_audio(audio_url), sampling_rate=16000, return_tensors="pt").input_features

# %%
# get special decoder tokens for the language
forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="english", task="transcribe")

# %%
forced_decoder_ids

# %%
input_features.shape

# %%
# perform inference
predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
predicted_ids.shape

# %%
# decode the IDs to text
transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
transcription

# %%
# decode the IDs to text with special tokens
transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=False)
transcription

# %%
def get_transcription_whisper(audio_path, model, processor, language="english", skip_special_tokens=True):
  # resample from whatever the audio sampling rate to 16000
  speech = load_audio(audio_path)
  # get the input features
  input_features = processor(speech, return_tensors="pt", sampling_rate=16000).input_features
  # get special decoder tokens for the language
  forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
  # perform inference
  predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
  # decode the IDs to text
  transcription = processor.batch_decode(predicted_ids, skip_special_tokens=skip_special_tokens)[0]
  return transcription

# %%
arabic_transcription = get_transcription_whisper("https://datasets-server.huggingface.co/assets/arabic_speech_corpus/--/clean/train/0/audio/audio.wav",
                          whisper_model,
                          whisper_processor,
                          language="arabic",
                          skip_special_tokens=True)
arabic_transcription

# %%
spanish_transcription = get_transcription_whisper("https://www.lightbulblanguages.co.uk/resources/sp-audio/cual-es-la-fecha-cumple.mp3",
                          whisper_model,
                          whisper_processor,
                          language="spanish",
                          skip_special_tokens=True)
spanish_transcription

# %%
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE 
# supported languages
TO_LANGUAGE_CODE 

# %% [markdown]
# # Transcribe your Voice

# %%
!git clone -q --depth 1 https://github.com/snakers4/silero-models

%cd silero-models

# %%
from IPython.display import Audio, display, clear_output
from colab_utils import record_audio
import ipywidgets as widgets
from scipy.io import wavfile
import numpy as np


record_seconds =   20#@param {type:"number", min:1, max:10, step:1}
sample_rate = 16000

def _record_audio(b):
  clear_output()
  audio = record_audio(record_seconds)
  display(Audio(audio, rate=sample_rate, autoplay=True))
  wavfile.write('recorded.wav', sample_rate, (32767*audio).numpy().astype(np.int16))

button = widgets.Button(description="Record Speech")
button.on_click(_record_audio)
display(button)

# %%
print("Whisper:", get_transcription_whisper("recorded.wav", whisper_model, whisper_processor))
print("Wav2vec2:", get_transcription_wav2vec2("recorded.wav", wav2vec2_model, wav2vec2_processor))


