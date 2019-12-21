import random
import os
import string
import secrets

# generate random integer between a and b (including a and b)
randint = random.randint(1, 500)
print("randint:", randint)

# generate random integer from range
randrange = random.randrange(0, 500, 5)
print("randrange:", randrange)

# get a random element from this list
choice = random.choice(["hello", "hi", "welcome", "bye", "see you"])
print("choice:", choice)

# get 5 random elements from 0 to 1000
choices = random.choices(range(1000), k=5)
print("choices:", choices)

# generate a random floating point number from 0.0 <= x <= 1.0
randfloat = random.random()
print("randfloat between 0.0 and 1.0:", randfloat)

# generate a random floating point number such that a <= x <= b
randfloat = random.uniform(5, 10)
print("randfloat between 5.0 and 10.0:", randfloat)

l = list(range(10))
print("Before shuffle:", l)
random.shuffle(l)
print("After shuffle:", l)

# generate a random string
randstring = ''.join(random.sample(string.ascii_letters, 16))
print("Random string with 16 characters:", randstring)

# crypto-safe byte generation
randbytes_crypto = os.urandom(16)
print("Random bytes for crypto use using os:", randbytes_crypto)

# or use this
randbytes_crypto = secrets.token_bytes(16)
print("Random bytes for crypto use using secrets:", randbytes_crypto)

# crypto-secure string generation
randstring_crypto = secrets.token_urlsafe(16)
print("Random strings for crypto use:", randstring_crypto)

# crypto-secure bits generation
randbits_crypto = secrets.randbits(16)
print("Random 16-bits for crypto use:", randbits_crypto)
