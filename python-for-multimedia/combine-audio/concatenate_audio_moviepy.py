from moviepy.editor import concatenate_audioclips, AudioFileClip


def concatenate_audio_moviepy(audio_clip_paths, output_path):
    """Concatenates several audio files into one audio file using MoviePy
    and save it to `output_path`. Note that extension (mp3, etc.) must be added to `output_path`"""
    clips = [AudioFileClip(c) for c in audio_clip_paths]
    final_clip = concatenate_audioclips(clips)
    final_clip.write_audiofile(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple Audio file combiner using MoviePy library in Python")
    parser.add_argument("-c", "--clips", nargs="+",
                        help="List of audio clip paths")
    parser.add_argument("-o", "--output", help="The output audio file, extension must be included (such as mp3, etc.)")
    args = parser.parse_args()
    concatenate_audio_moviepy(args.clips, args.output)
