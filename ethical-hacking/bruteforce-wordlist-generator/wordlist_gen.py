# Import the argparse module for handling command line arguments.
# Import the itertools module for generating combinations.
import argparse, itertools


# Define a function to generate a wordlist based on given parameters.
def generate_wordlist(characters, min_length, max_length, output_file):
    # Open the output file in write mode.
    with open(output_file, 'w') as file:
        # Iterate over the range of word lengths from min_length to max_length.
        for length in range(min_length, max_length + 1):
            # Generate all possible combinations of characters with the given length.
            for combination in itertools.product(characters, repeat=length):
                # Join the characters to form a word and write it to the file
                word = ''.join(combination)
                file.write(word + '\n')


# Create an ArgumentParser object for handling command line arguments.
parser = argparse.ArgumentParser(description="Generate a custom wordlist similar to crunch.")

# Define command line arguments.
parser.add_argument("-c", "--characters", type=str, default="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    help="Set of characters to include in the wordlist")
parser.add_argument("-min", "--min_length", type=int, default=4, help="Minimum length of the words")
parser.add_argument("-max", "--max_length", type=int, default=6, help="Maximum length of the words")
parser.add_argument("-o", "--output_file", type=str, default="custom_wordlist.txt", help="Output file name")

# Parse the command line arguments.
args = parser.parse_args()

# Call the generate_wordlist function with the provided arguments.
generate_wordlist(args.characters, args.min_length, args.max_length, args.output_file)

# Print a message indicating the wordlist has been generated and saved.
print(f"[+] Wordlist generated and saved to {args.output_file}")
