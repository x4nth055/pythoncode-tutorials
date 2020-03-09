import requests
import sys

# get the API KEY here: https://developers.google.com/custom-search/v1/overview
API_KEY = "<INSERT_YOUR_API_KEY_HERE>"
# get your Search Engine ID on your CSE control panel
SEARCH_ENGINE_ID = "<INSERT_YOUR_SEARCH_ENGINE_ID_HERE>"
# the search query you want, from the command line
query = sys.argv[1]
# constructing the URL
# doc: https://developers.google.com/custom-search/v1/using_rest
url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"

# make the API request
data = requests.get(url).json()
# get the result items
search_items = data.get("items")
# iterate over 10 results found
for i, search_item in enumerate(search_items, start=1):
    # get the page title
    title = search_item.get("title")
    # page snippet
    snippet = search_item.get("snippet")
    # alternatively, you can get the HTML snippet (bolded keywords)
    html_snippet = search_item.get("htmlSnippet")
    # extract the page url
    link = search_item.get("link")
    # print the results
    print("="*10, f"Result #{i}", "="*10)
    print("Title:", title)
    print("Description:", snippet)
    print("URL:", link, "\n")