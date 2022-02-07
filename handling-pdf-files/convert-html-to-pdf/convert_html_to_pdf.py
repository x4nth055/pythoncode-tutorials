import pdfkit

# directly from url
pdfkit.from_url("https://google.com", "google.pdf", verbose=True)
print("="*50)
# from file
pdfkit.from_file("webapp/index.html", "index.pdf", verbose=True, options={"enable-local-file-access": True})
print("="*50)
# from HTML content
pdfkit.from_string("<p><b>Python</b> is a great programming language.</p>", "string.pdf", verbose=True)
print("="*50)