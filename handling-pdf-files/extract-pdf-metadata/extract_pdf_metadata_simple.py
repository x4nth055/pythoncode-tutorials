import pikepdf
import sys

# get the target pdf file from the command-line arguments
pdf_filename = sys.argv[1]
# read the pdf file
pdf = pikepdf.Pdf.open(pdf_filename)
docinfo = pdf.docinfo
for key, value in docinfo.items():
    print(key, ":", value)
