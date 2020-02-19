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
- To play audio, there are 3 options (`audio_player_playsound.py` using [playsound](https://pypi.org/project/playsound/), `audio_player_pydub.py` using [pydub](https://github.com/jiaaro/pydub), `audio_player_pyaudio.py` using [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)), if you want to play `audio_file.mp3`::
    ```
    python audio_player_playsound.py audio_file.mp3
    ```