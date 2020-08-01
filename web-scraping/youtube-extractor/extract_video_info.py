from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs

# init session
session = HTMLSession()


def get_video_info(url):
    # download HTML code
    response = session.get(url)
    # execute Javascript
    response.html.render(sleep=1)
    # create beautiful soup object to parse HTML
    soup = bs(response.html.html, "html.parser")
    # open("index.html", "w").write(response.html.html)
    # initialize the result
    result = {}
    # video title
    result["title"] = soup.find("h1").text.strip()
    # video views (converted to integer)
    result["views"] = int(''.join([ c for c in soup.find("span", attrs={"class": "view-count"}).text if c.isdigit() ]))
    # video description
    result["description"] = soup.find("yt-formatted-string", {"class": "content"}).text
    # date published
    result["date_published"] = soup.find("div", {"id": "date"}).text[1:]
    # get the duration of the video
    result["duration"] = soup.find("span", {"class": "ytp-time-duration"}).text
    # get the video tags
    result["tags"] = ', '.join([ meta.attrs.get("content") for meta in soup.find_all("meta", {"property": "og:video:tag"}) ])
    # number of likes
    text_yt_formatted_strings = soup.find_all("yt-formatted-string", {"id": "text", "class": "ytd-toggle-button-renderer"})
    result["likes"] = int(''.join([ c for c in text_yt_formatted_strings[0].attrs.get("aria-label") if c.isdigit() ]))
    # number of dislikes
    result["dislikes"] = int(''.join([ c for c in text_yt_formatted_strings[1].attrs.get("aria-label") if c.isdigit() ]))

    # channel details
    channel_tag = soup.find("yt-formatted-string", {"class": "ytd-channel-name"}).find("a")
    # channel name
    channel_name = channel_tag.text
    # channel URL
    channel_url = f"https://www.youtube.com{channel_tag['href']}"
    # number of subscribers as str
    channel_subscribers = soup.find("yt-formatted-string", {"id": "owner-sub-count"}).text.strip()
    result['channel'] = {'name': channel_name, 'url': channel_url, 'subscribers': channel_subscribers}
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YouTube Video Data Extractor")
    parser.add_argument("url", help="URL of the YouTube video")

    args = parser.parse_args()
    # parse the video URL from command line
    url = args.url
    
    data = get_video_info(url)

    # print in nice format
    print(f"Title: {data['title']}")
    print(f"Views: {data['views']}")
    print(f"Published at: {data['date_published']}")
    print(f"Video Duration: {data['duration']}")
    print(f"Video tags: {data['tags']}")
    print(f"Likes: {data['likes']}")
    print(f"Dislikes: {data['dislikes']}")
    print(f"\nDescription: {data['description']}\n")
    print(f"\nChannel Name: {data['channel']['name']}")
    print(f"Channel URL: {data['channel']['url']}")
    print(f"Channel Subscribers: {data['channel']['subscribers']}")