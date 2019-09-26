import numpy as np
import pickle
import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import ModelCheckpoint



message = """
Please choose which model you want to generate text with:
1 - Alice's wonderland
2 - Python Code
"""
choice = int(input(message))
assert choice == 1 or choice == 2

if choice == 1:
    char2int = pickle.load(open("data/wonderland-char2int.pickle", "rb"))
    int2char = pickle.load(open("data/wonderland-int2char.pickle", "rb"))
elif choice == 2:
    char2int = pickle.load(open("data/python-char2int.pickle", "rb"))
    int2char = pickle.load(open("data/python-int2char.pickle", "rb"))

sequence_length = 100
n_unique_chars = len(char2int)

# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

if choice == 1:
    model.load_weights("results/wonderland-v2-0.75.h5")
elif choice == 2:
    model.load_weights("results/python-v2-0.30.h5")

seed = ""
print("Enter the seed, enter q to quit, maximum 100 characters:")
while True:
    result = input("")
    if result.lower() == "q":
        break
    seed += f"{result}\n"
seed = seed.lower()
n_chars = int(input("Enter number of characters you want to generate: "))

# generate 400 characters
generated = ""
for i in tqdm.tqdm(range(n_chars), "Generating text"):
    # make the input sequence
    X = np.zeros((1, sequence_length, n_unique_chars))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    # predict the next character
    predicted = model.predict(X, verbose=0)[0]
    # converting the vector to an integer
    next_index = np.argmax(predicted)
    # converting the integer to a character
    next_char = int2char[next_index]
    # add the character to results
    generated += next_char
    # shift seed and the predicted character
    seed = seed[1:] + next_char

print("Generated text:")
print(generated)