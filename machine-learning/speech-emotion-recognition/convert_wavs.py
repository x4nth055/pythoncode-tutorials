"""
A utility script used for converting audio samples to be 
suitable for feature extraction
"""

import os

def convert_audio(audio_path, target_path, remove=False):
    """This function sets the audio `audio_path` to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""

    os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
    # os.system(f"ffmpeg -i {audio_path} -ac 1 {target_path}")
    if remove:
        os.remove(audio_path)


def convert_audios(path, target_path, remove=False):
    """Converts a path of wav files to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
        and then put them into a new folder called `target_path`
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""

    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dirname = os.path.join(dirpath, dirname)
            target_dir = dirname.replace(path, target_path)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            if file.endswith(".wav"):
                # it is a wav file
                target_file = file.replace(path, target_path)
                convert_audio(file, target_file, remove=remove)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Convert ( compress ) wav files to 16MHz and mono audio channel ( 1 channel )
                                                    This utility helps for compressing wav files for training and testing""")
    parser.add_argument("audio_path", help="Folder that contains wav files you want to convert")
    parser.add_argument("target_path", help="Folder to save new wav files")
    parser.add_argument("-r", "--remove", type=bool, help="Whether to remove the old wav file after converting", default=False)

    args = parser.parse_args()
    audio_path = args.audio_path
    target_path = args.target_path

    if os.path.isdir(audio_path):
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
            convert_audios(audio_path, target_path, remove=args.remove)
    elif os.path.isfile(audio_path) and audio_path.endswith(".wav"):
        if not target_path.endswith(".wav"):
            target_path += ".wav"
        convert_audio(audio_path, target_path, remove=args.remove)
    else:
        raise TypeError("The audio_path file you specified isn't appropriate for this operation")
