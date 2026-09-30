import json
import unittest
from unittest.mock import Mock, patch

from extract_video_info import get_video_info


def make_html(player_response, contents=None):
    primary = {
        "title": {"runs": [{"text": "Test video"}]},
        "viewCount": {"videoViewCountRenderer": {"viewCount": {"simpleText": "1,234 views"}}},
        "dateText": {"simpleText": "Jan 1, 2026"},
        # An unrelated button appears first; the extractor must find the LIKE button.
        "topLevelButtons": [
            {"buttonViewModel": {"iconName": "SHARE", "title": "Share"}},
            {
                "buttonViewModel": {
                    "iconName": "LIKE",
                    "title": "19M",
                    "accessibilityText": "like this video along with 19,962,782 other people",
                }
            },
        ],
    }
    secondary = {
        "owner": {
            "videoOwnerRenderer": {
                "title": {"runs": [{"text": "Test channel"}]},
                "navigationEndpoint": {"browseEndpoint": {"browseId": "UC-test"}},
                "subscriberCountText": {"accessibility": {"accessibilityData": {"label": "10 subscribers"}}},
            }
        },
        "attributedDescription": {"content": "A test description"},
    }
    if contents is None:
        contents = [
            {"unrelatedRenderer": {}},
            {"videoPrimaryInfoRenderer": primary},
            {"videoSecondaryInfoRenderer": secondary},
        ]
    data = {
        "contents": {
            "twoColumnWatchNextResults": {
                "results": {"results": {"contents": contents}}
            }
        }
    }
    player = json.dumps(player_response) if isinstance(player_response, dict) else player_response
    return f"var ytInitialData = {json.dumps(data)}; var ytInitialPlayerResponse = {player};"


class ExtractVideoInfoTests(unittest.TestCase):
    def extract(self, html):
        response = Mock()
        response.text = html
        response.raise_for_status.return_value = None
        with patch("extract_video_info.requests.get", return_value=response) as get:
            result = get_video_info("https://www.youtube.com/watch?v=test")
        get.assert_called_once()
        self.assertEqual(get.call_args.kwargs["timeout"], 15)
        return result

    def test_exact_likes_and_hour_duration(self):
        result = self.extract(make_html({
            "videoDetails": {
                "lengthSeconds": "7322",
                "keywords": ["python", "test"],
            }
        }))

        self.assertEqual(result["likes"], "19962782")
        self.assertEqual(result["duration"], "2:02:02")
        self.assertEqual(result["tags"], "python, test")
        self.assertEqual(result["channel"]["name"], "Test channel")

    def test_live_stream_is_not_reported_as_elapsed_duration(self):
        result = self.extract(make_html({
            "videoDetails": {
                "lengthSeconds": "121601512",
                "isLiveContent": True,
            }
        }))

        self.assertEqual(result["duration"], "LIVE")

    def test_malformed_optional_player_response_keeps_page_metadata(self):
        result = self.extract(make_html("{invalid json}"))

        self.assertEqual(result["title"], "Test video")
        self.assertEqual(result["likes"], "19962782")
        self.assertEqual(result["duration"], "Duration not available")
        self.assertEqual(result["tags"], "No tags available")

    def test_missing_watch_page_structure_fails_with_clear_error(self):
        with self.assertRaisesRegex(Exception, "video may be unavailable"):
            self.extract(make_html({}, contents=[]))


if __name__ == "__main__":
    unittest.main()
