import requests
import urllib.parse as p
import sys

# get the API KEY here: https://developers.google.com/custom-search/v1/overview
API_KEY = "<INSERT_YOUR_API_KEY_HERE>"
# get your Search Engine ID on your CSE control panel
SEARCH_ENGINE_ID = "<INSERT_YOUR_SEARCH_ENGINE_ID_HERE>"
# target domain you want to track
target_domain = sys.argv[1]
# target keywords
query = sys.argv[2]

for page in range(1, 11):
    print("[*] Going for page:", page)
    # calculating start 
    start = (page - 1) * 10 + 1
    # make API request
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}"
    data = requests.get(url).json()
    search_items = data.get("items")
    # a boolean that indicates whether `target_domain` is found
    found = False
    for i, search_item in enumerate(search_items, start=1):
        # get the page title
        title = search_item.get("title")
        # page snippet
        snippet = search_item.get("snippet")
        # alternatively, you can get the HTML snippet (bolded keywords)
        html_snippet = search_item.get("htmlSnippet")
        # extract the page url
        link = search_item.get("link")
        # extract the domain name from the URL
        domain_name = p.urlparse(link).netloc
        if domain_name.endswith(target_domain):
            # get the page rank
            rank = i + start - 1
            print(f"[+] {target_domain} is found on rank #{rank} for keyword: '{query}'")
            print("[+] Title:", title)
            print("[+] Snippet:", snippet)
            print("[+] URL:", link)
            # target domain is found, exit out of the program
            found = True
            break
    if found:
        break
