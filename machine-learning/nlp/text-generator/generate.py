import numpy as np
import pickle
import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import ModelCheckpoint

# seed = "do not try to"

char2int = pickle.load(open("data/wonderland-char2int.pickle", "rb"))
int2char = pickle.load(open("data/wonderland-int2char.pickle", "rb"))

sequence_length = 100
n_unique_chars = len(char2int)

# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

model.load_weights("results/wonderland-v2-0.75.h5")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Text generator that was trained on Alice's Adventures in the Wonderland book.")
    parser.add_argument("seed", help="Seed text to start with, can be any english text, but it's preferable you take from the book itself.")
    parser.add_argument("-n", "--n-chars", type=int, dest="n_chars", help="Number of characters to generate, default is 200.", default=200)
    args = parser.parse_args()

    n_chars = args.n_chars
    seed = args.seed

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