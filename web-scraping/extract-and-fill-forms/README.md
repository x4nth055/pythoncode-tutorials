# [How to Extract and Submit Web Forms from a URL using Python](https://www.thepythoncode.com/article/extracting-and-submitting-web-page-forms-in-python)
To run this:
- `pip3 install -r requirements.txt`
- To extract forms, use `form_extractor.py`:
    ```
    python form_extractor.py https://wikipedia.org
    ```
- To extract and submit forms, use `form_submitter.py`:
    ```
    python form_submitter.py https://wikipedia.org
    ```
    This will extract the first form (you can change that in the code) and prompt the user for each non-hidden input field, and then submits the form and loads the respond HTML in your default web browser, try it out!