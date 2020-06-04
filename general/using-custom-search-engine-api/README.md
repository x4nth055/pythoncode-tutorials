# [How to Use Google Custom Search Engine API in Python](https://www.thepythoncode.com/article/use-google-custom-search-engine-api-in-python)
To run this:
- `pip3 install -r requirements.txt`
- You need to setup a CSE account, check [the tutorial](https://www.thepythoncode.com/article/use-google-custom-search-engine-api-in-python) on how you can set up one. 
- Change `API_KEY` and `SEARCH_ENGINE_ID` variables to yours, and then:
    ```
    python search_engine.py "python"
    ```
    This will use the query "python" to search for results, here is a cropped output:
    ```
    ========== Result #1 ==========
    Title: Welcome to Python.org
    Description: The official home of the Python Programming Language.
    URL: https://www.python.org/

    ========== Result #2 ==========
    Title: The Python Tutorial — Python 3.8.2 documentation
    Description: It has efficient high-level data structures and a simple but effective approach to 
    object-oriented programming. Python's elegant syntax and dynamic typing,
    together ...
    URL: https://docs.python.org/3/tutorial/

    ========== Result #3 ==========
    Title: Download Python | Python.org
    Description: Looking for Python with a different OS? Python for Windows, Linux/UNIX, Mac OS     
    X, Other. Want to help test development versions of Python? Prereleases ...
    URL: https://www.python.org/downloads/
    <..SNIPPED..>
    ```
- You can specify the page number, let's get 3rd result page for instance:
    ```
    python search_engine.py "python" 3
    ```
    Here is a **truncated output**:
    ```
    ========== Result #21 ==========
    Title: Python Tutorial - Tutorialspoint
    Description: Python is a general-purpose interpreted, interactive, object-oriented, and high-  
    level programming language. It was created by Guido van Rossum during 1985-
    ...
    URL: https://www.tutorialspoint.com/python/index.htm

    ========== Result #22 ==========
    Title: Google Python Style Guide
    Description: Python is the main dynamic language used at Google. This style guide is a list of 
    dos and don'ts for Python programs. To help you format code correctly, we've ...
    URL: http://google.github.io/styleguide/pyguide.html

    ========== Result #23 ==========
    Title: Individual Edition | Anaconda
    Description: Open Source Anaconda Individual Edition is the world's most popular Python        
    distribution platform with over 20 million users worldwide. You can trust in…
    URL: https://www.anaconda.com/products/individual
    <..SNIPPED..>
    ```