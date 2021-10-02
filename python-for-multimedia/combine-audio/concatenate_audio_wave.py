import wave

def concatenate_audio_wave(audio_clip_paths, output_path):
    """Concatenates several audio files into one audio file using Python's built-in wav module
    and save it to `output_path`. Note that extension (wav) must be added to `output_path`"""
    data = []
    for clip in audio_clip_paths:
        w = wave.open(clip, "rb")
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    output = wave.open(output_path, "wb")
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple Audio file combiner using wave module in Python")
    parser.add_argument("-c", "--clips", nargs="+",
                        help="List of audio clip paths")
    parser.add_argument("-o", "--output", help="The output audio file, extension (wav) must be included")
    args = parser.parse_args()
    concatenate_audio_wave(args.clips, args.output)