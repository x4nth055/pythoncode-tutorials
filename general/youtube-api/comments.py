from utils import youtube_authenticate, get_video_id_by_url, get_channel_id_by_url


def get_comments(youtube, **kwargs):
    return youtube.commentThreads().list(
        part="snippet",
        **kwargs
    ).execute()

        

if __name__ == "__main__":
    # authenticate to YouTube API
    youtube = youtube_authenticate()
    # URL can be a channel or a video, to extract comments
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw&ab_channel=jawed"
    if "watch" in url:
        # that's a video
        video_id = get_video_id_by_url(url)
        params = {
            'videoId': video_id, 
            'maxResults': 2,
            'order': 'relevance', # default is 'time' (newest)
        }
    else:
        # should be a channel
        channel_id = get_channel_id_by_url(url)
        params = {
            'allThreadsRelatedToChannelId': channel_id, 
            'maxResults': 2,
            'order': 'relevance', # default is 'time' (newest)
        }
    # get the first 2 pages (2 API requests)
    n_pages = 2
    for i in range(n_pages):
        # make API call to get all comments from the channel (including posts & videos)
        response = get_comments(youtube, **params)
        items = response.get("items")
        # if items is empty, breakout of the loop
        if not items:
            break
        for item in items:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            updated_at = item["snippet"]["topLevelComment"]["snippet"]["updatedAt"]
            like_count = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
            comment_id = item["snippet"]["topLevelComment"]["id"]
            print(f"""\
            Comment: {comment}
            Likes: {like_count}
            Updated At: {updated_at}
            ==================================\
            """)
        if "nextPageToken" in response:
            # if there is a next page
            # add next page token to the params we pass to the function
            params["pageToken"] =  response["nextPageToken"]
        else:
            # must be end of comments!!!!
            break
        print("*"*70)
