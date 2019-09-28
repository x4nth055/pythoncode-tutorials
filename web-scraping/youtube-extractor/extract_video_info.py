import requests
from bs4 import BeautifulSoup as bs


def get_video_info(url):
    # download HTML code
    content = requests.get(url)
    # create beautiful soup object to parse HTML
    soup = bs(content.content, "html.parser")
    # initialize the result
    result = {}
    # video title
    result['title'] = soup.find("span", attrs={"class": "watch-title"}).text.strip()
    # video views (converted to integer)
    result['views'] = int(soup.find("div", attrs={"class": "watch-view-count"}).text[:-6].replace(",", ""))
    # video description
    result['description'] = soup.find("p", attrs={"id": "eow-description"}).text
    # date published
    result['date_published'] = soup.find("strong", attrs={"class": "watch-time-text"}).text
    # number of likes as integer
    result['likes'] = int(soup.find("button", attrs={"title": "I like this"}).text.replace(",", ""))
    # number of dislikes as integer
    result['dislikes'] = int(soup.find("button", attrs={"title": "I dislike this"}).text.replace(",", ""))
    # channel details
    channel_tag = soup.find("div", attrs={"class": "yt-user-info"}).find("a")
    # channel name
    channel_name = channel_tag.text
    # channel URL
    channel_url = f"https://www.youtube.com{channel_tag['href']}"
    # number of subscribers as str
    channel_subscribers = soup.find("span", attrs={"class": "yt-subscriber-count"}).text.strip()
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
    print(f"\nDescription: {data['description']}\n")
    print(data['date_published'])
    print(f"Likes: {data['likes']}")
    print(f"Dislikes: {data['dislikes']}")
    print(f"\nChannel Name: {data['channel']['name']}")
    print(f"Channel URL: {data['channel']['url']}")
    print(f"Channel Subscribers: {data['channel']['subscribers']}")