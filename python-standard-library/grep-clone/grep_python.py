# Import the necessary libraries.
import re, sys
from colorama import init, Fore

# Initialize colorama.
init()

# Grep function.
def grep(pattern, filename):
    try:
        found_match = False
        with open(filename, 'r') as file:
            for line in file:
                if re.search(pattern, line):
                    # Print matching lines in green.
                    print(Fore.GREEN + line.strip() + "\n") # We are including new lines to enhance readability.
                    found_match = True
        if not found_match:
            # Print message in red if no content is found.
            print(Fore.RED + f"No content found matching the pattern '{pattern}'.")
    except FileNotFoundError:
        # Print error message in red if the file is not found.
        print(Fore.RED + f"File '{filename}' not found.")


if len(sys.argv) != 3:
    # Print usage message in red if the number of arguments is incorrect.
    print(Fore.RED + "Usage: python grep_python.py <pattern> <filename>")
    sys.exit(1)

pattern = sys.argv[1]
filename = sys.argv[2]
grep(pattern, filename)
