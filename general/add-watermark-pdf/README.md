# [How to Watermark PDF Files in Python](https://www.thepythoncode.com/article/watermark-in-pdf-using-python)
To run this:
- `pip3 install -r requirements.txt`
- ```python pdf_watermarker.py --help```

**Output:**
```

Available Options

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Enter the path of the file or the folder to process
  -a {watermark,unwatermark}, --action {watermark,unwatermark}
                        Choose whether to watermark or to unwatermark
  -m {RAM,HDD}, --mode {RAM,HDD}
                        Choose whether to process on the hard disk drive or in memory
  -w WATERMARK_TEXT, --watermark_text WATERMARK_TEXT
                        Enter a valid watermark text
  -p PAGES, --pages PAGES
                        Enter the pages to consider e.g.: [2,4]
```
- To add a watermark with any text on `lorem-ipsum.pdf` file and output it as `watermarked_lorem-ipsum.pdf`:
    ```
    python pdf_watermarker.py -i lorem-ipsum.pdf -a watermark -w "text here" -o watermarked_lorem-ipsum.pdf
    ```