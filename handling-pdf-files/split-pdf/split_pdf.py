import os
from pikepdf import Pdf

# a dictionary mapping PDF file to original PDF's page range
file2pages = {
    0: [0, 9], # 1st splitted PDF file will contain the pages from 0 to 9 (9 is not included)
    1: [9, 11], # 2nd splitted PDF file will contain the pages from 9 (9 is included) to 11
    2: [11, 100], # 3rd splitted PDF file will contain the pages from 11 until the end or until the 100th page (if exists)
}

# the target PDF document to split
filename = "bert-paper.pdf"
# load the PDF file
pdf = Pdf.open(filename)
# make the new splitted PDF files
new_pdf_files = [ Pdf.new() for i in file2pages ]
# the current pdf file index
new_pdf_index = 0
# iterate over all PDF pages
for n, page in enumerate(pdf.pages):
    if n in list(range(*file2pages[new_pdf_index])):
        # add the `n` page to the `new_pdf_index` file
        new_pdf_files[new_pdf_index].pages.append(page)
        print(f"[*] Assigning Page {n} to the file {new_pdf_index}")
    else:
        # make a unique filename based on original file name plus the index
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}-{new_pdf_index}.pdf"
        # save the PDF file
        new_pdf_files[new_pdf_index].save(output_filename)
        print(f"[+] File: {output_filename} saved.")
        # go to the next file
        new_pdf_index += 1
        # add the `n` page to the `new_pdf_index` file
        new_pdf_files[new_pdf_index].pages.append(page)
        print(f"[*] Assigning Page {n} to the file {new_pdf_index}")

# save the last PDF file
name, ext = os.path.splitext(filename)
output_filename = f"{name}-{new_pdf_index}.pdf"
new_pdf_files[new_pdf_index].save(output_filename)
print(f"[+] File: {output_filename} saved.")