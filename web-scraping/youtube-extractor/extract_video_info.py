import sys
import requests
import re
import json
import argparse

def _find_button_by_icon(obj, icon_name):
    """Find the first nested button view model with the requested icon."""
    if isinstance(obj, dict):
        if obj.get('iconName') == icon_name and 'title' in obj:
            return obj
        for value in obj.values():
            button = _find_button_by_icon(value, icon_name)
            if button is not None:
                return button
    elif isinstance(obj, list):
        for item in obj:
            button = _find_button_by_icon(item, icon_name)
            if button is not None:
                return button
    return None


def _format_duration(total_seconds):
    """Format a duration as M:SS or H:MM:SS."""
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{total_seconds // 60}:{seconds:02d}"


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
        response = requests.get(url, headers=headers, timeout=15)
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
        player_json = {}
        if player_match:
            try:
                player_json = json.loads(player_match.group(1))
            except json.JSONDecodeError:
                pass
        if not isinstance(player_json, dict):
            player_json = {}
        video_details = player_json.get('videoDetails', {})
        if not isinstance(video_details, dict):
            video_details = {}
        
        # Get the main content sections, and locate renderers by type rather than position.
        try:
            contents = data_json['contents']['twoColumnWatchNextResults']['results']['results']['contents']
        except (KeyError, TypeError):
            raise ValueError("Could not find video details; the video may be unavailable")
        if not isinstance(contents, list):
            raise ValueError("Could not find video details; the video may be unavailable")

        primary = next(
            (item['videoPrimaryInfoRenderer'] for item in contents
             if isinstance(item, dict) and 'videoPrimaryInfoRenderer' in item),
            None,
        )
        if primary is None:
            raise ValueError("Could not find video details; the video may be unavailable")

        # Video title, views, and publication date
        result["title"] = primary['title']['runs'][0]['text']
        result["views"] = primary['viewCount']['videoViewCountRenderer']['viewCount']['simpleText']
        result["date_published"] = primary['dateText']['simpleText']
        
        # Extract channel information from videoSecondaryInfoRenderer, if present.
        secondary = next(
            (item['videoSecondaryInfoRenderer'] for item in contents
             if isinstance(item, dict) and 'videoSecondaryInfoRenderer' in item),
            None,
        )
        if secondary:
            owner = secondary['owner']['videoOwnerRenderer']
            channel_name = owner['title']['runs'][0]['text']
            channel_id = owner['navigationEndpoint']['browseEndpoint']['browseId']
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
            channel_subscribers = owner['subscriberCountText']['accessibility']['accessibilityData']['label']

            result['channel'] = {
                'name': channel_name,
                'url': channel_url,
                'subscribers': channel_subscribers
            }
        else:
            result['channel'] = {
                'name': 'Channel not available',
                'url': 'Channel not available',
                'subscribers': 'Channel subscribers not available'
            }
        
        # Extract video description
        if secondary and 'attributedDescription' in secondary:
            description_runs = secondary['attributedDescription']['content']
            result["description"] = description_runs
        else:
            result["description"] = "Description not available"
        
        # Live streams report elapsed time in lengthSeconds, not media duration.
        if video_details.get('isLiveContent'):
            result["duration"] = "LIVE"
        else:
            try:
                duration_s = int(video_details['lengthSeconds'])
            except (KeyError, TypeError, ValueError):
                duration_s = None

            if duration_s is not None:
                result["duration"] = _format_duration(duration_s)
            else:
                duration_match = re.search(r'"approxDurationMs":"(\d+)"', response.text)
                if duration_match:
                    duration_ms = int(duration_match.group(1))
                    result["duration"] = _format_duration(duration_ms // 1000)
                else:
                    result["duration"] = "Duration not available"
        
        # Extract video tags (they live in ytInitialPlayerResponse.videoDetails.keywords,
        # NOT in ytInitialData.metadata like the old code assumed)
        video_tags = video_details.get('keywords', [])
        result["tags"] = ', '.join(video_tags) if video_tags else "No tags available"
        
        # Extract the exact count from the LIKE button's accessibility label when available.
        result["likes"] = "Likes count not available"
        button = _find_button_by_icon(primary, 'LIKE')
        if button:
            accessibility_text = button.get('accessibilityText', '')
            count_match = re.search(r'([\d,]+)\s+other people', accessibility_text, re.IGNORECASE)
            if count_match:
                result["likes"] = count_match.group(1).replace(',', '')
            else:
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
