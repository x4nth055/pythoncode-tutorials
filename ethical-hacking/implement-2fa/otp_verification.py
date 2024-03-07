# Program 2: Verify TOTP Code with Google Authenticator
import pyotp


def simulate_authentication(key):
    # Simulate the process of authenticating with a TOTP code.
    totp = pyotp.TOTP(key)
    print("Enter the code from your Google Authenticator app to complete authentication.")
    user_input = input("Enter Code: ")
    if totp.verify(user_input):
        print("Authentication successful!")
    else:
        print("Authentication failed. Please try again with the right key.")


# Main Code
# The key should be the same one generated and used to create the QR code in Program 1
user_key = open("2fa.txt").read()  # Reading the key from the file generated in Program 1 (otp_qrcode_and_key.py)
simulate_authentication(user_key)
