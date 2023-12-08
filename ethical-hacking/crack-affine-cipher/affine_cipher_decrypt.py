# Import the needed libraries.
import string
from colorama import Fore, init

# Initialise colorama.
init()


# Function to get Euclidean Algorithm.
def extended_gcd(a, b):
    """
    Extended Euclidean Algorithm to find the greatest common divisor
    and coefficients x, y such that ax + by = gcd(a, b).
    """
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = extended_gcd(b % a, a)
        return (g, y - (b // a) * x, x)


# Function to get the modular Inverse
def modular_inverse(a, m):
    """
    Compute the modular multiplicative inverse of a modulo m.
    Raises an exception if the modular inverse does not exist.
    """
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m


# Function to decrypt our message.
def affine_decrypt(ciphertext, a, b):
    """
    Decrypt a message encrypted with the Affine Cipher using
    the given key components a and b.
    """
    alphabet = string.ascii_uppercase
    m = len(alphabet)
    plaintext = ''

    # Compute the modular multiplicative inverse of a.
    a_inv = modular_inverse(a, m)

    # Iterate through each character in the ciphertext.
    for char in ciphertext:
        # Check if the character is in the alphabet
        if char in alphabet:
            # If it's an alphabet letter, decrypt it.
            # Find the index of the character in the alphabet.
            c = alphabet.index(char)
            # Apply the decryption formula: a_inv * (c - b) mod m.
            p = (a_inv * (c - b)) % m
            # Append the decrypted character to the plaintext.
            plaintext += alphabet[p]
        else:
            # If the character is not in the alphabet, keep it unchanged.
            plaintext += char

    # Return the decrypted plaintext.
    return plaintext


# Function to peform brute force attack.
def affine_brute_force(ciphertext):
    """
    Brute-force attack to find possible keys for an Affine Cipher
    and print potential decryptions for manual inspection.
    """
    alphabet = string.ascii_uppercase
    m = len(alphabet)

    # Iterate through possible values for a.
    for a in range(1, m):
        # Ensure a and m are coprime.
        if extended_gcd(a, m)[0] == 1:
            # Iterate through possible values for b.
            for b in range(0, m):
                # Decrypt using the current key.
                decrypted_text = affine_decrypt(ciphertext, a, b)

                # Print potential decryption for manual inspection.
                print(f"Key (a={a}, b={b}): {decrypted_text}")


ciphertext = input(f"{Fore.GREEN}[?] Enter Message to decrypt: ")

# Perform a brute-force attack to find potential decrypted message.
affine_brute_force(ciphertext)
