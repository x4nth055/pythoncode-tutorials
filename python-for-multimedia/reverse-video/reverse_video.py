from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import os
from datetime import timedelta, datetime
from glob import glob
from tqdm import tqdm
import shutil

# i.e if video of duration 30 seconds, saves 10 frame per second = 300 frames saved in total
SAVING_FRAMES_PER_SECOND = 30

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def extract_frames(video_file, verbose=1):
    # load the video clip
    video_clip = VideoFileClip(video_file)
    # make a folder by the name of the video file
    filename, _ = os.path.splitext(video_file)
    if not os.path.isdir(filename):
        os.mkdir(filename)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    iteration = np.arange(0, video_clip.duration, step)
    if verbose:
        iteration = tqdm(iteration, desc="Extracting video frames")
    # iterate over each possible frame
    for current_duration in iteration:
        # format the file name and save it
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration)).replace(":", "-")
        frame_filename = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
        # save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)
    return filename, video_clip.fps



def reverse_video(frames_path, video_fps, remove_extracted_frames=True):
    frame_files = glob(os.path.join(frames_path, "*"))
    # sort by duration in descending order
    frame_files = sorted(frame_files, key=lambda d: datetime.strptime(d.split("frame")[1], "%H-%M-%S.%f.jpg"), reverse=True)
    # calculate the FPS, getting the minimum between the original FPS and the parameter we set
    saving_frames_per_second = min(video_fps, SAVING_FRAMES_PER_SECOND)
    if saving_frames_per_second == 0:
        # if the parameter is set to 0, automatically set it to the original video fps
        saving_frames_per_second = video_fps
    print("Saving the video with FPS:", saving_frames_per_second)
    # load the frames into a image sequence clip (MoviePy)
    image_sequence_clip = ImageSequenceClip(frame_files, fps=saving_frames_per_second)
    # write the video file to disk
    output_filename = f"{frames_path}-inverted.mp4"
    image_sequence_clip.write_videofile(output_filename)
    if remove_extracted_frames:
        # if set to True, then remove the folder that contain the extracted frames
        shutil.rmtree(frames_path)



if __name__ == "__main__":
    import sys
    video_file = sys.argv[1]
    frames_folder_path, video_fps = extract_frames(video_file)
    reverse_video(frames_folder_path, video_fps=video_fps)
    
