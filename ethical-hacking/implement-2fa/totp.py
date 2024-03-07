import pyotp

# Generate a random key. You can also set to a variable e.g key = "CodingFleet"
key = pyotp.random_base32()
# Make Time based OTPs from the key.
totp = pyotp.TOTP(key)

# Print current key.
print(totp.now())

# Enter OTP for verification
input_code = input("Enter your OTP:")
# Verify OTP
print(totp.verify(input_code))