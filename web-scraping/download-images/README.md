# [How to Download All Images from a Web Page in Python](https://www.thepythoncode.com/article/download-web-page-images-python)
To run this:
- `pip3 install -r requirements.txt`
- 
    ```
    python download_images.py --help
    ```
    **Output:**
    ```
    usage: download_images.py [-h] [-p PATH] url

    This script downloads all images from a web page

    positional arguments:
    url                   The URL of the web page you want to download images

    optional arguments:
    -h, --help            show this help message and exit
    -p PATH, --path PATH  The Directory you want to store your images, default
                            is the domain of URL passed
    ```
- If you want to download all images from https://www.thepythoncode.com/topic/web-scraping for example:
    ```
    python download_images.py https://www.thepythoncode.com/topic/web-scraping
    ```
    A new folder `www.thepythoncode.com` will be created automatically that contains all the images of that web page.
- If you want to download images from javascript-driven websites, consider using `download_images_js.py` script instead (it accepts the same parameters)
