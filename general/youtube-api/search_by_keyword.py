from utils import (
    youtube_authenticate,  
    get_video_details,
    print_video_infos,
    search
)


if __name__ == "__main__":
    # authenticate to YouTube API
    youtube = youtube_authenticate()
    # search for the query 'python' and retrieve 2 items only
    response = search(youtube, q="python", maxResults=2)
    items = response.get("items")
    for item in items:
        # get the video ID
        video_id = item["id"]["videoId"]
        # get the video details
        video_response = get_video_details(youtube, id=video_id)
        # print the video details
        print_video_infos(video_response)
        print("="*50)