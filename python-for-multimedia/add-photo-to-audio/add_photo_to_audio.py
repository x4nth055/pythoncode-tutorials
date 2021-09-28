from moviepy.editor import AudioFileClip, ImageClip


def add_static_image_to_audio(image_path, audio_path, output_path):
    """Create and save a video file to `output_path` after 
    combining a static image that is located in `image_path` 
    with an audio file in `audio_path`"""
    # create the audio clip object
    audio_clip = AudioFileClip(audio_path)
    # create the image clip object
    image_clip = ImageClip(image_path)
    # use set_audio method from image clip to combine the audio with the image
    video_clip = image_clip.set_audio(audio_clip)
    # specify the duration of the new clip to be the duration of the audio clip
    video_clip.duration = audio_clip.duration
    # set the FPS to 1
    video_clip.fps = 1
    # write the resuling video clip
    video_clip.write_videofile(output_path)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple Python script to add a static image to an audio to make a video")
    parser.add_argument("image", help="The image path")
    parser.add_argument("audio", help="The audio path")
    parser.add_argument("output", help="The output video file path")
    args = parser.parse_args()
    add_static_image_to_audio(args.image, args.audio, args.output)