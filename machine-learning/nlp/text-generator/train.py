import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from string import punctuation

# commented because already downloaded
# import requests
# content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
# open("data/wonderland.txt", "w", encoding="utf-8").write(content)

# read the data
text = open("data/wonderland.txt", encoding="utf-8").read()
# remove caps
text = text.lower()
# remove punctuation
text = text.translate(str.maketrans("", "", punctuation))
# print some stats
n_chars = len(text)
unique_chars = ''.join(sorted(set(text)))
print("unique_chars:", unique_chars)
n_unique_chars = len(unique_chars)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(unique_chars)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(unique_chars)}

# save these dictionaries for later generation
pickle.dump(char2int, open("wonderland-char2int.pickle", "wb"))
pickle.dump(int2char, open("wonderland-int2char.pickle", "wb"))

# hyper parameters
sequence_length = 100
step = 1
batch_size = 128
epochs = 40

sentences = []
y_train = []
for i in range(0, len(text) - sequence_length, step):
    sentences.append(text[i: i + sequence_length])
    y_train.append(text[i+sequence_length])
print("Number of sentences:", len(sentences))

# vectorization
X = np.zeros((len(sentences), sequence_length, n_unique_chars))
y = np.zeros((len(sentences), n_unique_chars))

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char2int[char]] = 1
        y[i, char2int[y_train[i]]] = 1

print("X.shape:", X.shape)

# building the model
# model = Sequential([
#     LSTM(128, input_shape=(sequence_length, n_unique_chars)),
#     Dense(n_unique_chars, activation="softmax"),
# ])

# a better model (slower to train obviously)
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

# model.load_weights("results/wonderland-v2-2.48.h5")

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

if not os.path.isdir("results"):
    os.mkdir("results")

checkpoint = ModelCheckpoint("results/wonderland-v2-{loss:.2f}.h5", verbose=1)

# train the model
model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
