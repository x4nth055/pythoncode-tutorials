import fitz
import argparse
import sys
import os
from pprint import pprint

def get_arguments():
    parser = argparse.ArgumentParser(
        description="A Python script to extract text from PDF documents.")
    parser.add_argument("file", help="Input PDF file")
    parser.add_argument("-p", "--pages", nargs="*", type=int,
                        help="The pages to extract, default is all")
    parser.add_argument("-o", "--output-file", default=sys.stdout,
                        help="Output file to write text. default is standard output")
    parser.add_argument("-b", "--by-page", action="store_true",
                        help="Whether to output text by page. If not specified, all text is joined and will be written together")
    # parse the arguments from the command-line
    args = parser.parse_args()

    input_file = args.file
    pages = args.pages
    by_page = args.by_page
    output_file = args.output_file
    # print the arguments, just for logging purposes
    pprint(vars(args))
    # load the pdf file
    pdf = fitz.open(input_file)
    if not pages:
        # if pages is not set, default is all pages of the input PDF document
        pages = list(range(pdf.pageCount))
    # we make our dictionary that maps each pdf page to its corresponding file
    # based on passed arguments
    if by_page:
        if output_file is not sys.stdout:
            # if by_page and output_file are set, open all those files
            file_name, ext = os.path.splitext(output_file)
            output_files = { pn: open(f"{file_name}-{pn}{ext}", "w") for pn in pages }
        else:
            # if output file is standard output, do not open
            output_files = { pn: output_file for pn in pages }
    else:
        if output_file is not sys.stdout:
            # a single file, open it
            output_file = open(output_file, "w")
            output_files = { pn: output_file for pn in pages }
        else:
            # if output file is standard output, do not open
            output_files = { pn: output_file for pn in pages }

    # return the parsed and processed arguments
    return {
        "pdf": pdf,
        "output_files": output_files,
        "pages": pages,
    }


def extract_text(**kwargs):
    # extract the arguments
    pdf          = kwargs.get("pdf")
    output_files = kwargs.get("output_files")
    pages        = kwargs.get("pages")
    # iterate over pages
    for pg in range(pdf.pageCount):
        if pg in pages:
            # get the page object
            page = pdf[pg]
            # extract the text of that page and split by new lines '\n'
            page_lines = page.get_text().splitlines()
            # get the output file
            file = output_files[pg]
            # get the number of lines
            n_lines = len(page_lines)
            for line in page_lines:
                # remove any whitespaces in the end & beginning of the line
                line = line.strip()
                # print the line to the file/stdout
                print(line, file=file)
            print(f"[*] Wrote {n_lines} lines in page {pg}")
                
    # close the files
    for pn, f in output_files.items():
        if f is not sys.stdout:
            f.close()
            
            
if __name__ == "__main__":
    # get the arguments
    kwargs = get_arguments()
    # extract text from the pdf document
    extract_text(**kwargs)
