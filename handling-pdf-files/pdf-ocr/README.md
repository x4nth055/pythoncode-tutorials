# [How to Extract Text from Images in PDF Files with Python](https://www.thepythoncode.com/article/extract-text-from-images-or-scanned-pdf-python)
To run this:
- `pip3 install -r requirements.txt`
-
    ```
    $ python pdf_ocr.py --help
    ```

    **Output:**
    ```
    usage: pdf_ocr.py [-h] -i INPUT_PATH [-a {Highlight,Redact}] [-s SEARCH_STR] [-p PAGES] [-g]

    Available Options

    optional arguments:
    -h, --help            show this help message and exit
    -i INPUT_PATH, --input-path INPUT_PATH
                            Enter the path of the file or the folder to process
    -a {Highlight,Redact}, --action {Highlight,Redact}
                            Choose to highlight or to redact
    -s SEARCH_STR, --search-str SEARCH_STR
                            Enter a valid search string
    -p PAGES, --pages PAGES
                            Enter the pages to consider in the PDF file, e.g. (0,1)
    -g, --generate-output
                            Generate text content in a CSV file
    ```
- To extract text from scanned image in `image.pdf` file:
    ```
    $ python pdf_ocr.py -s "BERT" -i image.pdf -o output.pdf --generate-output -a Highlight
    ```
    Passing `-s` to search for the keyword, `-i` is to pass the input file, `-o` is to pass output PDF file, `--generate-output` or `-g` to generate CSV file containing all extract text from all images in the PDF file, and `-a` for specifiying the action to perform in the output PDF file, "Highlight" will highlight the target keyword, you can also pass "Redact" to redact the text instead.