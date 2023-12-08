# Import necessary libraries.
import string
from colorama import init, Fore

# Initialise colorama.
init()


# Function to perform Affine Cipher encryption.
def affine_encryption(plaintext, a, b):
    # Define the uppercase alphabet.
    alphabet = string.ascii_uppercase
    # Get the length of the alphabet
    m = len(alphabet)
    # Initialize an empty string to store the ciphertext.
    ciphertext = ''

    # Iterate through each character in the plaintext.
    for char in plaintext:
        # Check if the character is in the alphabet.
        if char in alphabet:
            # If it's an alphabet letter, encrypt it.
            # Find the index of the character in the alphabet.
            p = alphabet.index(char)
            # Apply the encryption formula: (a * p + b) mod m.
            c = (a * p + b) % m
            # Append the encrypted character to the ciphertext.
            ciphertext += alphabet[c]
        else:
            # If the character is not in the alphabet, keep it unchanged.
            ciphertext += char

    # Return the encrypted ciphertext.
    return ciphertext


# Define the plaintext and key components.
plaintext = input(f"{Fore.GREEN}[?] Enter text to encrypt: ")
a = 3
b = 10

# Call the affine_encrypt function with the specified parameters.
encrypted_text = affine_encryption(plaintext, a, b)

# Print the original plaintext, the key components, and the encrypted text.
print(f"{Fore.MAGENTA}[+] Plaintext: {plaintext}")
print(f"{Fore.GREEN}[+] Encrypted Text: {encrypted_text}")
