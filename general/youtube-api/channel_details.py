from utils import (
    youtube_authenticate, 
    get_channel_id_by_url, 
    get_channel_details,
    get_video_details,
    print_video_infos
)


def get_channel_videos(youtube, **kwargs):
    return youtube.search().list(
        **kwargs
    ).execute()
        

if __name__ == "__main__":
    # authenticate to YouTube API
    youtube = youtube_authenticate()
    channel_url = "https://www.youtube.com/channel/UC8butISFwT-Wl7EV0hUK0BQ"
    # get the channel ID from the URL
    channel_id = get_channel_id_by_url(youtube, channel_url)
    # get the channel details
    response = get_channel_details(youtube, id=channel_id)
    # extract channel infos
    snippet = response["items"][0]["snippet"]
    statistics = response["items"][0]["statistics"]
    channel_country = snippet["country"]
    channel_description = snippet["description"]
    channel_creation_date = snippet["publishedAt"]
    channel_title = snippet["title"]
    channel_subscriber_count = statistics["subscriberCount"]
    channel_video_count = statistics["videoCount"]
    channel_view_count  = statistics["viewCount"]
    print(f"""
    Title: {channel_title}
    Published At: {channel_creation_date}
    Description: {channel_description}
    Country: {channel_country}
    Number of videos: {channel_video_count}
    Number of subscribers: {channel_subscriber_count}
    Total views: {channel_view_count}
    """)
    # the following is grabbing channel videos
    # number of pages you want to get
    n_pages = 2
    # counting number of videos grabbed
    n_videos = 0
    next_page_token = None
    for i in range(n_pages):
        params = {
            'part': 'snippet',
            'q': '',
            'channelId': channel_id,
            'type': 'video',
        }
        if next_page_token:
            params['pageToken'] = next_page_token
        res = get_channel_videos(youtube, **params)
        channel_videos = res.get("items")
        for video in channel_videos:
            n_videos += 1
            video_id = video["id"]["videoId"]
            # easily construct video URL by its ID
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_response = get_video_details(youtube, id=video_id)
            print(f"================Video #{n_videos}================")
            # print the video details
            print_video_infos(video_response)
            print(f"Video URL: {video_url}")
            print("="*40)
        # if there is a next page, then add it to our parameters
        # to proceed to the next page
        if "nextPageToken" in res:
            next_page_token = res["nextPageToken"]