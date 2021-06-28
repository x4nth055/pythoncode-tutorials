from PyPDF4 import PdfFileReader, PdfFileWriter
from PyPDF4.pdf import ContentStream
from PyPDF4.generic import TextStringObject, NameObject
from PyPDF4.utils import b_
import os
import argparse
from io import BytesIO
from typing import Tuple
# Import the reportlab library
from reportlab.pdfgen import canvas
# The size of the page supposedly A4
from reportlab.lib.pagesizes import A4
# The color of the watermark
from reportlab.lib import colors

PAGESIZE = A4
FONTNAME = 'Helvetica-Bold'
FONTSIZE = 40
# using colors module
# COLOR = colors.lightgrey
# or simply RGB
# COLOR = (190, 190, 190)
COLOR = colors.red
# The position attributes of the watermark
X = 250
Y = 10
# The rotation angle in order to display the watermark diagonally if needed
ROTATION_ANGLE = 45


def get_info(input_file: str):
    """
    Extracting the file info
    """
    # If PDF is encrypted the file metadata cannot be extracted
    with open(input_file, 'rb') as pdf_file:
        pdf_reader = PdfFileReader(pdf_file, strict=False)
        output = {
            "File": input_file, "Encrypted": ("True" if pdf_reader.isEncrypted else "False")
        }
        if not pdf_reader.isEncrypted:
            info = pdf_reader.getDocumentInfo()
            num_pages = pdf_reader.getNumPages()
            output["Author"] = info.author
            output["Creator"] = info.creator
            output["Producer"] = info.producer
            output["Subject"] = info.subject
            output["Title"] = info.title
            output["Number of pages"] = num_pages
    # To Display collected metadata
    print("## File Information ##################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in output.items()))
    print("######################################################################")
    return True, output


def get_output_file(input_file: str, output_file: str):
    """
    Check whether a temporary output file is needed or not
    """
    input_path = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    # If output file is empty -> generate a temporary output file
    # If output file is equal to input_file -> generate a temporary output file
    if not output_file or input_file == output_file:
        tmp_file = os.path.join(input_path, 'tmp_' + input_filename)
        return True, tmp_file
    return False, output_file


def create_watermark(wm_text: str):
    """
    Creates a watermark template.
    """
    if wm_text:
        # Generate the output to a memory buffer
        output_buffer = BytesIO()
        # Default Page Size = A4
        c = canvas.Canvas(output_buffer, pagesize=PAGESIZE)
        # you can also add image instead of text
        # c.drawImage("logo.png", X, Y, 160, 160)
        # Set the size and type of the font
        c.setFont(FONTNAME, FONTSIZE)
        # Set the color
        if isinstance(COLOR, tuple):
            color = (c/255 for c in COLOR)
            c.setFillColorRGB(*color)
        else:
            c.setFillColor(COLOR)
        # Rotate according to the configured parameter
        c.rotate(ROTATION_ANGLE)
        # Position according to the configured parameter
        c.drawString(X, Y, wm_text)
        c.save()
        return True, output_buffer
    return False, None


def save_watermark(wm_buffer, output_file):
    """
    Saves the generated watermark template to disk
    """
    with open(output_file, mode='wb') as f:
        f.write(wm_buffer.getbuffer())
    f.close()
    return True


def watermark_pdf(input_file: str, wm_text: str, pages: Tuple = None):
    """
    Adds watermark to a pdf file.
    """
    result, wm_buffer = create_watermark(wm_text)
    if result:
        wm_reader = PdfFileReader(wm_buffer)
        pdf_reader = PdfFileReader(open(input_file, 'rb'), strict=False)
        pdf_writer = PdfFileWriter()
        try:
            for page in range(pdf_reader.getNumPages()):
                # If required to watermark specific pages not all the document pages
                if pages:
                    if str(page) not in pages:
                        continue
                page = pdf_reader.getPage(page)
                page.mergePage(wm_reader.getPage(0))
                pdf_writer.addPage(page)
        except Exception as e:
            print("Exception = ", e)
            return False, None, None

        return True, pdf_reader, pdf_writer


def unwatermark_pdf(input_file: str, wm_text: str, pages: Tuple = None):
    """
    Removes watermark from the pdf file.
    """
    pdf_reader = PdfFileReader(open(input_file, 'rb'), strict=False)
    pdf_writer = PdfFileWriter()
    for page in range(pdf_reader.getNumPages()):
        # If required for specific pages
        if pages:
            if str(page) not in pages:
                continue
        page = pdf_reader.getPage(page)
        # Get the page content
        content_object = page["/Contents"].getObject()
        content = ContentStream(content_object, pdf_reader)
        # Loop through all the elements page elements
        for operands, operator in content.operations:
            # Checks the TJ operator and replaces the corresponding string operand (Watermark text) with ''
            if operator == b_("Tj"):
                text = operands[0]
                if isinstance(text, str) and text.startswith(wm_text):
                    operands[0] = TextStringObject('')
        page.__setitem__(NameObject('/Contents'), content)
        pdf_writer.addPage(page)
    return True, pdf_reader, pdf_writer


def watermark_unwatermark_file(**kwargs):
    input_file = kwargs.get('input_file')
    wm_text = kwargs.get('wm_text')
    # watermark   -> Watermark
    # unwatermark -> Unwatermark
    action = kwargs.get('action')
    # HDD -> Temporary files are saved on the Hard Disk Drive and then deleted
    # RAM -> Temporary files are saved in memory and then deleted.
    mode = kwargs.get('mode')
    pages = kwargs.get('pages')
    temporary, output_file = get_output_file(
        input_file, kwargs.get('output_file'))
    if action == "watermark":
        result, pdf_reader, pdf_writer = watermark_pdf(
            input_file=input_file, wm_text=wm_text, pages=pages)
    elif action == "unwatermark":
        result, pdf_reader, pdf_writer = unwatermark_pdf(
            input_file=input_file, wm_text=wm_text, pages=pages)
    # Completed successfully
    if result:
        # Generate to memory
        if mode == "RAM":
            output_buffer = BytesIO()
            pdf_writer.write(output_buffer)
            pdf_reader.stream.close()
            # No need to create a temporary file in RAM Mode
            if temporary:
                output_file = input_file
            with open(output_file, mode='wb') as f:
                f.write(output_buffer.getbuffer())
            f.close()
        elif mode == "HDD":
            # Generate to a new file on the hard disk
            with open(output_file, 'wb') as pdf_output_file:
                pdf_writer.write(pdf_output_file)
            pdf_output_file.close()

            pdf_reader.stream.close()
            if temporary:
                if os.path.isfile(input_file):
                    os.replace(output_file, input_file)
                output_file = input_file


def watermark_unwatermark_folder(**kwargs):
    """
    Watermarks all PDF Files within a specified path
    Unwatermarks all PDF Files within a specified path
    """
    input_folder = kwargs.get('input_folder')
    wm_text = kwargs.get('wm_text')
    # Run in recursive mode
    recursive = kwargs.get('recursive')
    # watermark   -> Watermark
    # unwatermark -> Unwatermark
    action = kwargs.get('action')
    # HDD -> Temporary files are saved on the Hard Disk Drive and then deleted
    # RAM -> Temporary files are saved in memory and then deleted.
    mode = kwargs.get('mode')
    pages = kwargs.get('pages')
    # Loop though the files within the input folder.
    for foldername, dirs, filenames in os.walk(input_folder):
        for filename in filenames:
            # Check if pdf file
            if not filename.endswith('.pdf'):
                continue
            # PDF File found
            inp_pdf_file = os.path.join(foldername, filename)
            print("Processing file:", inp_pdf_file)
            watermark_unwatermark_file(input_file=inp_pdf_file, output_file=None,
                                       wm_text=wm_text, action=action, mode=mode, pages=pages)
        if not recursive:
            break


def is_valid_path(path):
    """
    Validates the path inputted and checks whether it is a file path or a folder path
    """
    if not path:
        raise ValueError(f"Invalid Path")
    if os.path.isfile(path):
        return path
    elif os.path.isdir(path):
        return path
    else:
        raise ValueError(f"Invalid Path {path}")


def parse_args():
    """
    Get user command line parameters
    """
    parser = argparse.ArgumentParser(description="Available Options")
    parser.add_argument('-i', '--input_path', dest='input_path', type=is_valid_path,
                        required=True, help="Enter the path of the file or the folder to process")
    parser.add_argument('-a', '--action', dest='action', choices=[
                        'watermark', 'unwatermark'], type=str, default='watermark',
                        help="Choose whether to watermark or to unwatermark")
    parser.add_argument('-m', '--mode', dest='mode', choices=['RAM', 'HDD'], type=str,
                        default='RAM', help="Choose whether to process on the hard disk drive or in memory")
    parser.add_argument('-w', '--watermark_text', dest='watermark_text',
                        type=str, required=True, help="Enter a valid watermark text")
    parser.add_argument('-p', '--pages', dest='pages', type=tuple,
                        help="Enter the pages to consider e.g.: [2,4]")
    path = parser.parse_known_args()[0].input_path
    if os.path.isfile(path):
        parser.add_argument('-o', '--output_file', dest='output_file',
                            type=str, help="Enter a valid output file")
    if os.path.isdir(path):
        parser.add_argument('-r', '--recursive', dest='recursive', default=False, type=lambda x: (
            str(x).lower() in ['true', '1', 'yes']), help="Process Recursively or Non-Recursively")
    # To Porse The Command Line Arguments
    args = vars(parser.parse_args())
    # To Display The Command Line Arguments
    print("## Command Arguments #################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in args.items()))
    print("######################################################################")
    return args


if __name__ == '__main__':
    # Parsing command line arguments entered by user
    args = parse_args()
    # If File Path
    if os.path.isfile(args['input_path']):
        # Extracting File Info
        get_info(input_file=args['input_path'])
        # Encrypting or Decrypting a File
        watermark_unwatermark_file(
            input_file=args['input_path'], wm_text=args['watermark_text'], action=args[
                'action'], mode=args['mode'], output_file=args['output_file'], pages=args['pages']
        )
    # If Folder Path
    elif os.path.isdir(args['input_path']):
        # Encrypting or Decrypting a Folder
        watermark_unwatermark_folder(
            input_folder=args['input_path'], wm_text=args['watermark_text'],
            action=args['action'], mode=args['mode'], recursive=args['recursive'], pages=args['pages']
        )
