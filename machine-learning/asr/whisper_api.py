import openai

# API key
openai.api_key = "<API_KEY>"

def get_openai_api_transcription(audio_filename):
    # open the audio file
    with open(audio_filename, "rb") as audio_file:
        # transcribe the audio file
        transcription = openai.Audio.transcribe("whisper-1", audio_file) # whisper-1 is the model name
    return transcription


if __name__ == "__main__":
    transcription = get_openai_api_transcription("7601-291468-0006.wav")
    print(transcription.get("text"))