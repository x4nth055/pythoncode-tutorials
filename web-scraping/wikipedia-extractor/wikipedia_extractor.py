import wikipedia

# print the summary of what python is
print(wikipedia.summary("Python Programming Language"))

# search for a term
result = wikipedia.search("Neural networks")
print("Result search of 'Neural networks':", result)

# get the page: Neural network
page = wikipedia.page(result[0])

# get the title of the page
title = page.title

# get the categories of the page
categories = page.categories

# get the whole wikipedia page text (content)
content = page.content

# get all the links in the page
links = page.links

# get the page references
references = page.references

# summary
summary = page.summary

# print info
print("Page content:\n", content, "\n")
print("Page title:", title, "\n")
print("Categories:", categories, "\n")
print("Links:", links, "\n")
print("References:", references, "\n")
print("Summary:", summary, "\n")

# changing language
# for a list of available languages, 
# check http://meta.wikimedia.org/wiki/List_of_Wikipedias link.
language = "es"
wikipedia.set_lang(language)
# get a page and print the summary in the new language
print(f"Summary of web scraping in {language}:", wikipedia.page("Web Scraping").summary)

