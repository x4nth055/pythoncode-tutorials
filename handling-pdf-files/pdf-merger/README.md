# [How to Merge PDF Files in Python](https://www.thepythoncode.com/article/merge-pdf-files-in-python)
To run this:
- `pip3 install -r requirements.txt`
-
    ```
    $ python pdf_merger.py --help
    ```
    **Output:**
    ```
    usage: pdf_merger.py [-h] -i [INPUT_FILES [INPUT_FILES ...]] [-p [PAGE_RANGE [PAGE_RANGE ...]]] -o OUTPUT_FILE [-b BOOKMARK]

    Available Options

    optional arguments:
    -h, --help            show this help message and exit
    -i [INPUT_FILES [INPUT_FILES ...]], --input_files [INPUT_FILES [INPUT_FILES ...]]
                            Enter the path of the files to process
    -p [PAGE_RANGE [PAGE_RANGE ...]], --page_range [PAGE_RANGE [PAGE_RANGE ...]]
                            Enter the pages to consider e.g.: (0,2) -> First 2 pages
    -o OUTPUT_FILE, --output_file OUTPUT_FILE
                            Enter a valid output file
    -b BOOKMARK, --bookmark BOOKMARK
                            Bookmark resulting file
    ```
- To merge `bert-paper.pdf` with `letter.pdf` into a new `combined.pdf`:
    ```
    $ python pdf_merger.py -i bert-paper.pdf,letter.pdf -o combined.pdf
    ```