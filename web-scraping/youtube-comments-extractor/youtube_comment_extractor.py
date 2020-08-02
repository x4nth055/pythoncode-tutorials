import requests
import json
import time

# Code is partially grabbed from this repository:
# https://github.com/egbertbouman/youtube-comment-downloader

def search_dict(partial, key):
    """
    A handy function that searches for a specific `key` in a `data` dictionary/list
    """
    if isinstance(partial, dict):
        for k, v in partial.items():
            if k == key:
                # found the key, return the value
                yield v
            else:
                # value of the dict may be another dict, so we search there again
                for o in search_dict(v, key):
                    yield o
    elif isinstance(partial, list):
        # if the passed data is a list
        # iterate over it & search for the key at the items in the list
        for i in partial:
            for o in search_dict(i, key):
                yield o


def find_value(html, key, num_sep_chars=2, separator='"'):
    # define the start position by the position of the key + 
    # length of key + separator length (usually : and ")
    start_pos = html.find(key) + len(key) + num_sep_chars
    # the end position is the position of the separator (such as ")
    # starting from the start_pos
    end_pos = html.find(separator, start_pos)
    # return the content in this range
    return html[start_pos:end_pos]


def get_comments(url):
    session = requests.Session()
    # make the request
    res = session.get(url)
    # extract the XSRF token
    xsrf_token = find_value(res.text, "XSRF_TOKEN", num_sep_chars=3)
    # parse the YouTube initial data in the <script> tag
    data_str = find_value(res.text, 'window["ytInitialData"] = ', num_sep_chars=0, separator="\n").rstrip(";")
    # convert to Python dictionary instead of plain text string
    data = json.loads(data_str)
    # search for the ctoken & continuation parameter fields
    for r in search_dict(data, "itemSectionRenderer"):
        pagination_data = next(search_dict(r, "nextContinuationData"))
        if pagination_data:
            # if we got something, break out of the loop,
            # we have the data we need
            break

    continuation_tokens = [(pagination_data['continuation'], pagination_data['clickTrackingParams'])]

    while continuation_tokens:
        # keep looping until continuation tokens list is empty (no more comments)
        continuation, itct = continuation_tokens.pop()
    
        # construct params parameter (the ones in the URL)
        params = {
            "action_get_comments": 1,
            "pbj": 1,
            "ctoken": continuation,
            "continuation": continuation,
            "itct": itct,
        }

        # construct POST body data, which consists of the XSRF token
        data = {
            "session_token": xsrf_token,
        }

        # construct request headers
        headers = {
            "x-youtube-client-name": "1",
            "x-youtube-client-version": "2.20200731.02.01"
        }

        # make the POST request to get the comments data
        response = session.post("https://www.youtube.com/comment_service_ajax", params=params, data=data, headers=headers)
        # convert to a Python dictionary
        comments_data = json.loads(response.text)

        for comment in search_dict(comments_data, "commentRenderer"):
            # iterate over loaded comments and yield useful info
            yield {
                "commentId": comment["commentId"],
                "text": ''.join([c['text'] for c in comment['contentText']['runs']]),
                "time": comment['publishedTimeText']['runs'][0]['text'],
                "isLiked": comment["isLiked"],
                "likeCount": comment["likeCount"],
                # "replyCount": comment["replyCount"],
                'author': comment.get('authorText', {}).get('simpleText', ''),
                'channel': comment['authorEndpoint']['browseEndpoint']['browseId'],
                'votes': comment.get('voteCount', {}).get('simpleText', '0'),
                'photo': comment['authorThumbnail']['thumbnails'][-1]['url'],
                "authorIsChannelOwner": comment["authorIsChannelOwner"],
            }

        # load continuation tokens for next comments (ctoken & itct)
        continuation_tokens = [(next_cdata['continuation'], next_cdata['clickTrackingParams'])
                         for next_cdata in search_dict(comments_data, 'nextContinuationData')] + continuation_tokens

        # avoid heavy loads with popular videos
        time.sleep(0.1)
    




if __name__ == "__main__":
    # from pprint import pprint
    # url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    # for count, comment in enumerate(get_comments(url)):
    #     if count == 3:
    #         break
    #     pprint(comment)
    #     print("="*50)
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Simple YouTube Comment extractor")
    parser.add_argument("url", help="The YouTube video full URL")
    parser.add_argument("-l", "--limit", type=int, help="Number of maximum comments to extract, helpful for longer videos")
    parser.add_argument("-o", "--output", help="Output JSON file, e.g data.json")

    # parse passed arguments
    args = parser.parse_args()
    limit = args.limit
    output = args.output
    url = args.url

    from pprint import pprint
    for count, comment in enumerate(get_comments(url)):
        if limit and count >= limit:
            # break out of the loop when we exceed limit specified
            break
        if output:
            # write comment as JSON to a file
            with open(output, "a") as f:
                # begin writing, adding an opening brackets
                if count == 0:
                    f.write("[")
                f.write(json.dumps(comment, ensure_ascii=False) + ",")
        else:
            pprint(comment)
            print("="*50)
    print("total comments extracted:", count)
    if output:
        # remove the last comma ','
        with open(output, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            f.truncate()
        # add "]" to close the list in the end of the file
        with open(output, "a") as f:
            print("]", file=f)
