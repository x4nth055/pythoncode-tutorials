# [How to Extract YouTube Comments in Python](https://www.thepythoncode.com/article/extract-youtube-comments-in-python)
To run this:
- `pip3 install -r requirements.txt`
- ```
    python youtube_comment_extractor.py --help
    ```
    **Output:**
    ```
    usage: youtube_comment_extractor.py [-h] [-l LIMIT] [-o OUTPUT] url

    Simple YouTube Comment extractor

    positional arguments:
    url                   The YouTube video full URL

    optional arguments:
    -h, --help            show this help message and exit
    -l LIMIT, --limit LIMIT
                            Number of maximum comments to extract, helpful for
                            longer videos
    -o OUTPUT, --output OUTPUT
                            Output JSON file, e.g data.json
    ```
- To download the latest 50 comments from https://www.youtube.com/watch?v=jNQXAC9IVRw and save them to `data.json`:
    ```
    python youtube_comment_extractor.py https://www.youtube.com/watch?v=jNQXAC9IVRw --limit 50 --output data.json
    ```