from pydub.silence import split_on_silence
from pydub import AudioSegment
from whisper_api import get_openai_api_transcription
import os

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription_on_silence(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_file(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        transcription = get_openai_api_transcription(chunk_filename)
        print(f"{chunk_filename}: {transcription.get('text')}")
        whole_text += " " + transcription.get("text")
    # return the text for all chunks detected
    return whole_text


# a function that splits the audio file into fixed interval chunks
# and applies speech recognition
def get_large_audio_transcription_fixed_interval(path, minutes=5):
    """
    Splitting the large audio file into 5-minute chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_file(path)  
    # split the audio file into chunks
    chunk_length_ms = int(1000 * 60 * minutes) # convert to milliseconds
    chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]
    folder_name = "audio-fixed-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        transcription = get_openai_api_transcription(chunk_filename)
        print(f"{chunk_filename}: {transcription.get('text')}")
        whole_text += " " + transcription.get("text")
    # return the text for all chunks detected
    return whole_text



if __name__ == "__main__":
    # print("\nFull text:", get_large_audio_transcription_fixed_interval("032.mp3", minutes=1))
    print("\nFull text:", get_large_audio_transcription_on_silence("7601-291468-0006.wav"))