# [How to Convert Speech to Text in Python](https://www.thepythoncode.com/article/using-speech-recognition-to-convert-speech-to-text-python)

This folder contains the original `SpeechRecognition` examples and a modern 2026 transcription script.

## Modern script

`speech_to_text_2026.py` supports:

- OpenAI `gpt-4o-transcribe` / `gpt-4o-mini-transcribe`
- Faster-Whisper local/offline transcription
- Groq Whisper transcription
- long-audio chunking
- microphone recording
- SRT subtitle export

Install modern dependencies:

```bash
pip install -U openai faster-whisper groq sounddevice scipy
```

For audio/video conversion and long-file chunking, install FFmpeg too.

Examples:

```bash
# Local/offline transcription
python speech_to_text_2026.py 16-122828-0002.wav --engine faster-whisper --model small --language en

# OpenAI transcription; requires OPENAI_API_KEY
python speech_to_text_2026.py meeting.mp3 --engine openai --language en

# Cheaper OpenAI model
python speech_to_text_2026.py meeting.mp3 --engine openai --model gpt-4o-mini-transcribe --language en

# Groq Whisper; requires GROQ_API_KEY
python speech_to_text_2026.py meeting.mp3 --engine groq --language en

# Generate subtitles locally
python speech_to_text_2026.py video.mp4 --engine faster-whisper --model large-v3 --srt captions.srt

# Record 8 seconds from the microphone, then transcribe
python speech_to_text_2026.py --record 8 --engine faster-whisper --model small --language en
```

## Legacy examples

To run the older examples:

```bash
pip3 install -r requirements.txt
```

Recognize the text of an audio file named `16-122828-0002.wav`:

```bash
python recognizer.py 16-122828-0002.wav
```

Output:

```text
I believe you're just talking nonsense
```

Recognize text from your microphone after talking for 5 seconds:

```bash
python live_recognizer.py 5
```
