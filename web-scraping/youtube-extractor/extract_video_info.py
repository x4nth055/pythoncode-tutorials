import sys
import requests
from bs4 import BeautifulSoup
import re
import json
import argparse

def _find_key(obj, key):
    """Recursively search a nested dict/list structure for the first occurrence of `key`."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def get_video_info(url):
    """
    Extract video information from YouTube using modern approach
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        # Without this, YouTube serves the page in the visitor's locale (e.g. Arabic),
        # so views/dates/subscribers come back in the wrong language
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        # Download HTML code
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Initialize the result
        result = {}
        
        # Extract ytInitialData which contains the video/watch page information
        data_match = re.search(r'var ytInitialData = ({.*?});', response.text)
        if not data_match:
            raise Exception("Could not find ytInitialData in page")
            
        data_json = json.loads(data_match.group(1))
        
        # Extract ytInitialPlayerResponse (video metadata: keywords, duration, view count)
        player_match = re.search(r'var ytInitialPlayerResponse = ({.*?});', response.text)
        player_json = json.loads(player_match.group(1)) if player_match else {}
        video_details = player_json.get('videoDetails', {})
        
        # Get the main content sections
        contents = data_json['contents']['twoColumnWatchNextResults']['results']['results']['contents']
        
        # Extract video information from videoPrimaryInfoRenderer
        if 'videoPrimaryInfoRenderer' in contents[0]:
            primary = contents[0]['videoPrimaryInfoRenderer']
            
            # Video title
            result["title"] = primary['title']['runs'][0]['text']
            
            # Video views
            result["views"] = primary['viewCount']['videoViewCountRenderer']['viewCount']['simpleText']
            
            # Date published
            result["date_published"] = primary['dateText']['simpleText']
        
        # Extract channel information from videoSecondaryInfoRenderer
        secondary = None
        if 'videoSecondaryInfoRenderer' in contents[1]:
            secondary = contents[1]['videoSecondaryInfoRenderer']
            owner = secondary['owner']['videoOwnerRenderer']
            
            # Channel name
            channel_name = owner['title']['runs'][0]['text']
            
            # Channel ID
            channel_id = owner['navigationEndpoint']['browseEndpoint']['browseId']
            
            # Channel URL - FIXED with proper /channel/ path
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
            
            # Number of subscribers
            channel_subscribers = owner['subscriberCountText']['accessibility']['accessibilityData']['label']
            
            result['channel'] = {
                'name': channel_name, 
                'url': channel_url, 
                'subscribers': channel_subscribers
            }
        
        # Extract video description
        if secondary and 'attributedDescription' in secondary:
            description_runs = secondary['attributedDescription']['content']
            result["description"] = description_runs
        else:
            result["description"] = "Description not available"
        
        # Extract video duration from the player response (accurate, e.g. "19" seconds),
        # with a fallback regex on the page HTML
        if 'lengthSeconds' in video_details:
            duration_s = int(video_details['lengthSeconds'])
            minutes = duration_s // 60
            seconds = duration_s % 60
            result["duration"] = f"{minutes}:{seconds:02d}"
        else:
            duration_match = re.search(r'"approxDurationMs":"(\d+)"', response.text)
            if duration_match:
                duration_ms = int(duration_match.group(1))
                minutes = duration_ms // 60000
                seconds = (duration_ms % 60000) // 1000
                result["duration"] = f"{minutes}:{seconds:02d}"
            else:
                result["duration"] = "Duration not available"
        
        # Extract video tags (they live in ytInitialPlayerResponse.videoDetails.keywords,
        # NOT in ytInitialData.metadata like the old code assumed)
        video_tags = video_details.get('keywords', [])
        result["tags"] = ', '.join(video_tags) if video_tags else "No tags available"
        
        # Extract likes (2026 structure):
        # videoActions.menuRenderer.topLevelButtons[0].segmentedLikeDislikeButtonViewModel
        #   .likeButtonViewModel.likeButtonViewModel.toggleButtonViewModel
        #   .toggleButtonViewModel.defaultButtonViewModel.buttonViewModel.title
        result["likes"] = "Likes count not available"
        if 'videoPrimaryInfoRenderer' in contents[0]:
            button = _find_key(contents[0]['videoPrimaryInfoRenderer'], 'buttonViewModel')
            if button and button.get('iconName') == 'LIKE' and 'title' in button:
                result["likes"] = button['title'].replace('\xa0', ' ')
        
        # Dislikes are not published by YouTube anymore
        result["dislikes"] = "UNKNOWN"
        
        return result
        
    except Exception as e:
        raise Exception(f"Error extracting video info: {str(e)}")

if __name__ == "__main__":
    # Avoid UnicodeEncodeError crashes on Windows consoles (cp1252)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="YouTube Video Data Extractor")
    parser.add_argument("url", help="URL of the YouTube video")

    args = parser.parse_args()
    
    # parse the video URL from command line
    url = args.url
    
    try:
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
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: YouTube frequently changes its structure, so this script may need updates.")
