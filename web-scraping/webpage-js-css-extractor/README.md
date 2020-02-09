# [How to Extract Script and CSS Files from Web Pages in Python](https://www.thepythoncode.com/article/extract-web-page-script-and-css-files-in-python)
To run this:
- `pip3 install -r requirements.txt`
- Extracting `http://books.toscrape.com`'s CSS & Script files:
    ```
    python extractor.py http://books.toscrape.com/
    ```
    2 files will appear, one for javascript files (`javascript_files.txt`) and the other for CSS files (`css_files.txt`)