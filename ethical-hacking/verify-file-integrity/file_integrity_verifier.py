# Import necessary libraries.
import argparse, hashlib, sys

# Import functions init and Fore from the colorama library.
from colorama import init, Fore

# Initialize colorama to enable colored terminal text.
init()

# Define a function to calculate the SHA-256 hash of a file.
def calculate_hash(file_path):
    # Create a SHA-256 hash object.
    sha256_hash = hashlib.sha256()

    # Open the file in binary mode for reading (rb).
    with open(file_path, "rb") as file:
        # Read the file in 64KB chunks to efficiently handle large files.
        while True:
            data = file.read(65536)  # Read the file in 64KB chunks.
            if not data:
                break
            # Update the hash object with the data read from the file.
            sha256_hash.update(data)

    # Return the hexadecimal representation of the calculated hash.
    return sha256_hash.hexdigest()


# Define a function to verify the calculated hash against an expected hash.
def verify_hash(downloaded_file, expected_hash):
    # Calculate the hash of the downloaded file.
    calculated_hash = calculate_hash(downloaded_file)

    # Compare the calculated hash with the expected hash and return the result.
    return calculated_hash == expected_hash


# Create a parser for handling command-line arguments.
parser = argparse.ArgumentParser(description="Verify the hash of a downloaded software file.")

# Define two command-line arguments:
# -f or --file: Path to the downloaded software file (required).
# --hash: Expected hash value (required).
parser.add_argument("-f", "--file", dest="downloaded_file", required=True, help="Path to the downloaded software file")
parser.add_argument("--hash", dest="expected_hash", required=True, help="Expected hash value")

# Parse the command-line arguments provided when running the script.
args = parser.parse_args()

# Check if the required command-line arguments were provided.
if not args.downloaded_file or not args.expected_hash:
    # Print an error message in red using 'colorama'.
    print(f"{Fore.RED}[-] Please Specify the file to validate and its Hash.")
    # Exit the script.
    sys.exit()

# Check if the hash of the file is accurate by calling the verify_hash function.
if verify_hash(args.downloaded_file, args.expected_hash):
    # If the hash is accurate, print a success message in green.
    print(f"{Fore.GREEN}[+] Hash verification successful. The software is authentic.")
else:
    # If the hash does not match, print an error message in red.
    print(f"{Fore.RED}[-] Hash verification failed. The software may have been tampered with or is not authentic.")
