import pikepdf # pip3 install pikepdf

file = "1810.04805.pdf"
# file = "1710.05006.pdf"
pdf_file = pikepdf.Pdf.open(file)
urls = []
# iterate over PDF pages
for page in pdf_file.pages:
    for annots in page.get("/Annots"):
        uri = annots.get("/A").get("/URI")
        if uri is not None:
            print("[+] URL Found:", uri)
            urls.append(uri)

print("[*] Total URLs extracted:", len(urls))