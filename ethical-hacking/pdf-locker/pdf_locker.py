# Import the necessary libraries
import PyPDF2, getpass # getpass is for getting password with some level of security
from colorama import Fore, init

# Initialize colorama for colored output
init()


# Function to lock pdf
def lock_pdf(input_file, password):
    with open(input_file, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Create a PDF writer object
        pdf_writer = PyPDF2.PdfWriter()

        # Add all pages to the writer
        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])

        # Encrypt the PDF with the provided password
        pdf_writer.encrypt(password)

        # Write the encrypted content back to the original file
        with open(input_file, 'wb') as output_file:
            pdf_writer.write(output_file)


# Get user input
input_pdf = input("Enter the path to the PDF file: ")
password = getpass.getpass("Enter the password to lock the PDF: ")

# Lock the PDF using PyPDF2
print(f'{Fore.GREEN}[!] Please hold on for a few seconds..')
lock_pdf(input_pdf, password)

# Let the user know it's done
print(f"{Fore.GREEN}[+] PDF locked successfully.")
