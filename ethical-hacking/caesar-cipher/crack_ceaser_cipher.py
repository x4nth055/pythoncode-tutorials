# Import colorama for colorful text.
from colorama import Fore, init

init()


# Define a function for Caesar cipher encryption.
def implement_caesar_cipher(text, key, decrypt=False):
    # Initialize an empty string to store the result.
    result = ""

    # Iterate through each character in the input text.
    for char in text:
        # Check if the character is alphabetical.
        if char.isalpha():
            # Determine the shift value using the provided key (or its negation for decryption).
            shift = key if not decrypt else -key

            # Check if the character is lowercase
            if char.islower():
                # Apply the Caesar cipher encryption/decryption formula for lowercase letters.
                result += chr(((ord(char) - ord('a') + shift) % 26) + ord('a'))
            else:
                # Apply the Caesar cipher encryption/decryption formula for uppercase letters.
                result += chr(((ord(char) - ord('A') + shift) % 26) + ord('A'))
        else:
            # If the character is not alphabetical, keep it as is e.g. numbers, punctuation
            result += char

    # Return the result, which is the encrypted or decrypted text
    return result


# Define a function for cracking the Caesar cipher.
def crack_caesar_cipher(ciphertext):
    # Iterate through all possible keys (0 to 25) as there 26 alphabets.
    for key in range(26):
        # Call the caesar_cipher function with the current key to decrypt the text.
        decrypted_text = implement_caesar_cipher(ciphertext, key, decrypt=True)

        # Print the result, showing the decrypted text for each key
        print(f"{Fore.RED}Key {key}: {decrypted_text}")


# Initiate a continuous loop so the program keeps running.
while True:
    # Accept user input.
    encrypted_text = input(f"{Fore.GREEN}[?] Please Enter the text/message to decrypt: ")
    # Check if user does not specify anything.
    if not encrypted_text:
        print(f"{Fore.RED}[-] Please specify the text to decrypt.")
    else:
        crack_caesar_cipher(encrypted_text)


