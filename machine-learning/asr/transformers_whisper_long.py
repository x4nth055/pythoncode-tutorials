
from transformers import pipeline
import torch
import torchaudio

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# whisper_model_name = "openai/whisper-tiny.en" # English-only, ~ 151 MB
# whisper_model_name = "openai/whisper-base.en" # English-only, ~ 290 MB
# whisper_model_name = "openai/whisper-small.en" # English-only, ~ 967 MB
# whisper_model_name = "openai/whisper-medium.en" # English-only, ~ 3.06 GB
whisper_model_name = "openai/whisper-tiny" # multilingual, ~ 151 MB
# whisper_model_name = "openai/whisper-base" # multilingual, ~ 290 MB
# whisper_model_name = "openai/whisper-small" # multilingual, ~ 967 MB
# whisper_model_name = "openai/whisper-medium" # multilingual, ~ 3.06 GB
# whisper_model_name = "openai/whisper-large-v2" # multilingual, ~ 6.17 GB

def load_audio(audio_path):
  """Load the audio file & convert to 16,000 sampling rate"""
  # load our wav file
  speech, sr = torchaudio.load(audio_path)
  resampler = torchaudio.transforms.Resample(sr, 16000)
  speech = resampler(speech)
  return speech.squeeze()


def get_long_transcription_whisper(audio_path, pipe, return_timestamps=True, 
                                   chunk_length_s=10, stride_length_s=1):
    """Get the transcription of a long audio file using the Whisper model"""
    return pipe(load_audio(audio_path).numpy(), return_timestamps=return_timestamps,
                chunk_length_s=chunk_length_s, stride_length_s=stride_length_s)
    
    
if __name__ == "__main__":
    # initialize the pipeline
    pipe = pipeline("automatic-speech-recognition", 
                model=whisper_model_name, device=device)
    # get the transcription of a sample long audio file
    output = get_long_transcription_whisper(
        "7601-291468-0006.wav", pipe, chunk_length_s=10, stride_length_s=2)
    print(f"Transcription: {output}")
    print("="*50)
    for chunk in output["chunks"]:
        # print the timestamp and the text
        print(chunk["timestamp"], ":", chunk["text"])