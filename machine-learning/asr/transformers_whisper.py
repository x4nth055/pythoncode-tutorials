from transformers import WhisperProcessor, WhisperForConditionalGeneration
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

# load the model and the processor
whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)

def load_audio(audio_path):
  """Load the audio file & convert to 16,000 sampling rate"""
  # load our wav file
  speech, sr = torchaudio.load(audio_path)
  resampler = torchaudio.transforms.Resample(sr, 16000)
  speech = resampler(speech)
  return speech.squeeze()


def get_transcription_whisper(audio_path, model, processor, language="english", skip_special_tokens=True):
  # resample from whatever the audio sampling rate to 16000
  speech = load_audio(audio_path)
  # get the input features from the audio file
  input_features = processor(speech, return_tensors="pt", sampling_rate=16000).input_features.to(device)
  # get the forced decoder ids
  forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
  # print(forced_decoder_ids)
  # generate the transcription
  predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
  # decode the predicted ids
  transcription = processor.batch_decode(predicted_ids, skip_special_tokens=skip_special_tokens)[0]
  return transcription


if __name__ == "__main__":
    english_transcription = get_transcription_whisper("7601-291468-0006.wav",
                            whisper_model,
                            whisper_processor,
                            language="english",
                            skip_special_tokens=True)
    print("English transcription:", english_transcription)
    arabic_transcription = get_transcription_whisper("arabic-audio.wav",
                          whisper_model,
                          whisper_processor,
                          language="arabic",
                          skip_special_tokens=True)
    print("Arabic transcription:", arabic_transcription)
    spanish_transcription = get_transcription_whisper("cual-es-la-fecha-cumple.mp3",
                          whisper_model,
                          whisper_processor,
                          language="spanish",
                          skip_special_tokens=True)
    print("Spanish transcription:", spanish_transcription)
    