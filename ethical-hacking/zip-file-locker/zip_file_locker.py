# Import the necessary libraries.
import pyzipper, argparse, sys, re, getpass
from colorama import Fore, init

init()

# Define a function to get CLI commands.
def get_cli_arguments():
    parser = argparse.ArgumentParser(description="A program to lock a ZIP File.")
    # Collect user arguments.
    parser.add_argument('--zipfile', '-z', dest='zip_file', help='Specify the ZIP file to create or update.')
    parser.add_argument('--addfile', '-a', dest='add_files', nargs='+', help='Specify one or more files to add to the ZIP file(s).')

    # Parse the collected arguments.
    args = parser.parse_args()

    # Check if arguments are missing, print appropriate messages and exit the program.
    if not args.zip_file:
        parser.print_help()
        sys.exit()
    if not args.add_files:
        parser.print_help()
        sys.exit()

    return args

# Function to check password strength.
def check_password_strength(password):
    # Check for minimum length. In our case, 8.
    if len(password) < 8:
        return False

    # Check for at least one uppercase letter, one lowercase letter, and one digit.
    if not (re.search(r'[A-Z]', password) and re.search(r'[a-z]', password) and re.search(r'\d', password)):
        return False

    return True

# Call the arguments function.
arguments = get_cli_arguments()

# Get user password
password = getpass.getpass("[?] Enter your password > ")

# If password is weak, tell the user and exit the program.
if not check_password_strength(password):
    print(f"{Fore.RED}[-] Password is not strong enough. It should have at least 8 characters and contain at least one uppercase letter, one lowercase letter, and one digit.")
    sys.exit()

# Create a password-protected ZIP file.
with pyzipper.AESZipFile(arguments.zip_file, 'w', compression=pyzipper.ZIP_LZMA, encryption=pyzipper.WZ_AES) as zf:
    zf.setpassword(password.encode())

    # Add files to the ZIP file.
    for file_to_add in arguments.add_files:
        zf.write(file_to_add)

# Print a Success message.
print(f"{Fore.GREEN}[+] ZIP file is locked with a strong password.")
