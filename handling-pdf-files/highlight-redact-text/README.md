# [Highlighting Text in PDF with Python](https://www.thepythoncode.com/article/redact-and-highlight-text-in-pdf-with-python)
To run this:
- `pip3 install -r requirements.txt`
- 
    ```python pdf_highlighter.py --help```
    **Output:**
    ```
    usage: pdf_highlighter.py [-h] -i INPUT_PATH [-a {Redact,Frame,Highlight,Squiggly,Underline,Strikeout,Remove}] [-p PAGES]

    Available Options

    optional arguments:
    -h, --help            show this help message and exit
    -i INPUT_PATH, --input_path INPUT_PATH
                            Enter the path of the file or the folder to process
    -a {Redact,Frame,Highlight,Squiggly,Underline,Strikeout,Remove}, --action {Redact,Frame,Highlight,Squiggly,Underline,Strikeout,Remove}
                            Choose whether to Redact or to Frame or to Highlight or to Squiggly or to Underline or to Strikeout or to Remove
    -p PAGES, --pages PAGES
                            Enter the pages to consider e.g.: [2,4]
    ```