from utils import (
    youtube_authenticate,  
    get_video_id_by_url, 
    get_video_details,
    print_video_infos
)


if __name__ == "__main__":
    # authenticate to YouTube API
    youtube = youtube_authenticate()
    video_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw&ab_channel=jawed"
    # parse video ID from URL
    video_id = get_video_id_by_url(video_url)
    # make API call to get video info
    response = get_video_details(youtube, id=video_id)
    # print extracted video infos
    print_video_infos(response)