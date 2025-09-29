import requests
from bs4 import BeautifulSoup
import re
import json
import argparse

def get_video_info(url):
    """
    Extract video information from YouTube using modern approach
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Download HTML code
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Create beautiful soup object to parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Initialize the result
        result = {}
        
        # Extract ytInitialData which contains all the video information
        data_match = re.search(r'var ytInitialData = ({.*?});', response.text)
        if not data_match:
            raise Exception("Could not find ytInitialData in page")
            
        data_json = json.loads(data_match.group(1))
        
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
        
        # Try to extract video duration from player overlay
        # This is a fallback approach since the original method doesn't work
        duration_match = re.search(r'"approxDurationMs":"(\d+)"', response.text)
        if duration_match:
            duration_ms = int(duration_match.group(1))
            minutes = duration_ms // 60000
            seconds = (duration_ms % 60000) // 1000
            result["duration"] = f"{minutes}:{seconds:02d}"
        else:
            result["duration"] = "Duration not available"
        
        # Extract video tags if available
        video_tags = []
        if 'keywords' in data_json.get('metadata', {}).get('videoMetadataRenderer', {}):
            video_tags = data_json['metadata']['videoMetadataRenderer']['keywords']
        result["tags"] = ', '.join(video_tags) if video_tags else "No tags available"
        
        # Extract likes (modern approach)
        result["likes"] = "Likes count not available"
        result["dislikes"] = "UNKNOWN"  # YouTube no longer shows dislikes
        
        # Try to find likes in the new structure
        for content in contents:
            if 'compositeVideoPrimaryInfoRenderer' in content:
                composite = content['compositeVideoPrimaryInfoRenderer']
                if 'likeButton' in composite:
                    like_button = composite['likeButton']
                    if 'toggleButtonRenderer' in like_button:
                        toggle = like_button['toggleButtonRenderer']
                        if 'defaultText' in toggle:
                            default_text = toggle['defaultText']
                            if 'accessibility' in default_text:
                                accessibility = default_text['accessibility']
                                if 'accessibilityData' in accessibility:
                                    label = accessibility['accessibilityData']['label']
                                    if 'like' in label.lower():
                                        result["likes"] = label
        
        return result
        
    except Exception as e:
        raise Exception(f"Error extracting video info: {str(e)}")

if __name__ == "__main__":
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