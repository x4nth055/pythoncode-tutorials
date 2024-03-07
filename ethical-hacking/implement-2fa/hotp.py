import pyotp

# Set the key. A variable this time
key = 'Muhammad'
# Make a HMAC-based OTP
hotp = pyotp.HOTP(key)

# Print results
print(hotp.at(0))
print(hotp.at(1))
print(hotp.at(2))
print(hotp.at(3))

# Set counter
counter = 0
for otp in range(4):
    print(hotp.verify(input("Enter Code: "), counter))
    counter += 1

