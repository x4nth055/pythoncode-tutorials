import fitz # pip install PyMuPDF
import re

# a regular expression of URLs
url_regex = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
# extract raw text from pdf
# file = "1710.05006.pdf"
file = "1810.04805.pdf"
# open the PDF file
with fitz.open(file) as pdf:
    text = ""
    for page in pdf:
        # extract text of each PDF page
        text += page.getText()
urls = []
# extract all urls using the regular expression
for match in re.finditer(url_regex, text):
    url = match.group()
    print("[+] URL Found:", url)
    urls.append(url)
print("[*] Total URLs extracted:", len(urls))

