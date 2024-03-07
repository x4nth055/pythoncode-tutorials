# Program 1: Generate and Save TOTP Key and QR Code
import pyotp
import qrcode


def generate_otp_key():
    # Generate a random key for TOTP authentication.
    return pyotp.random_base32()


def generate_qr_code(key, account_name, issuer_name):
    # Generate a QR code for TOTP authentication.
    uri = pyotp.totp.TOTP(key).provisioning_uri(name=account_name, issuer_name=issuer_name)
    img = qrcode.make(uri)
    img.save('totp_qr.png')
    print("QR Code generated and saved as 'totp_qr.png'.")


# Main code.
# Generate user key.
user_key = generate_otp_key()
print("Your Two-Factor Authentication Key:", user_key)
# Save key to a file for reference purposes
with open('2fa.txt', 'w') as f:
    f.write(user_key)
# Generate QR Code.
generate_qr_code(user_key, 'Muhammad', 'CodingFleet.com')
