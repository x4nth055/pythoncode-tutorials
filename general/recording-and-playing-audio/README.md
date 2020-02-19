# [How to Play and Record Audio in Python](https://www.thepythoncode.com/article/play-and-record-audio-sound-in-python)
To run this:
- `pip3 install -r requirements.txt`
- To record audio:
    ```
    python audio_recorder.py --help
    ```
    **Output:**
    ```
    usage: audio_recorder.py [-h] [-o OUTPUT] [-d DURATION]

    an Audio Recorder using Python

    optional arguments:
    -h, --help            show this help message and exit
    -o OUTPUT, --output OUTPUT
                            Output file (with .wav)
    -d DURATION, --duration DURATION
                            Duration to record in seconds (can be float)
    ```
    For instance, you want to record 5 seconds and save it to `recorded.wav` file:
    ```
    python audio_recorder.py -d 5 -o recorded.wav
    ```
- To play audio, there are 3 options (`audio_player_1.py` using [playsound](), `audio_player_2.py` using [pydub](), `audio_player_3.py` using [pyaudio]()):
    ```
    python audio_player.py --help
    ```