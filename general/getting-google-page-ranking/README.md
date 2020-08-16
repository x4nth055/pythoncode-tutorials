# [How to Get Google Page Ranking in Python](https://www.thepythoncode.com/article/get-google-page-ranking-by-keyword-in-python)
To run this:
- `pip3 install -r requirements.txt`
- Setup CSE API and retrieve `API_KEY` and `SEARCH_ENGINE_ID` as shown in [this tutorial](https://www.thepythoncode.com/article/use-google-custom-search-engine-api-in-python) and replace them in `page_ranking.py` script.
- For instance, to get the page rank of `thepythoncode.com` of the keyword "google custom search engine api python":
    ```
    python page_ranking.py thepythoncode.com "google custom search engine api python"
    ```
    **Output:**
    ```
    [*] Going for page: 1
    [+] thepythoncode.com is found on rank #3 for keyword: 'google custom search engine api python'
    [+] Title: How to Use Google Custom Search Engine API in Python - Python ...
    [+] Snippet: 10 results ... Learning how to create your own Google Custom Search Engine and use its 
    Application Programming Interface (API) in Python.
    [+] URL: https://www.thepythoncode.com/article/use-google-custom-search-engine-api-in-python 
    ```