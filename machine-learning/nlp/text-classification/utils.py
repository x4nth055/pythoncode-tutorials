from tqdm import tqdm

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

from glob import glob
import random


def get_embedding_vectors(word_index, embedding_size=100):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    with open(f"data/glove.6B.{embedding_size}d.txt", encoding="utf8") as f:
        for line in tqdm(f, "Reading GloVe"):
            values = line.split()
            # get the word as the first word in the line
            word = values[0]
            if word in word_index:
                idx = word_index[word]
                # get the vectors as the remaining values in the line
                embedding_matrix[idx] = np.array(values[1:], dtype="float32")
    return embedding_matrix


def create_model(word_index, units=128, n_layers=1, cell=LSTM, bidirectional=False,
                embedding_size=100, sequence_length=100, dropout=0.3, 
                loss="categorical_crossentropy", optimizer="adam", 
                output_length=2):
    """
    Constructs a RNN model given its parameters
    """
    embedding_matrix = get_embedding_vectors(word_index, embedding_size)
    model = Sequential()
    # add the embedding layer
    model.add(Embedding(len(word_index) + 1,
              embedding_size,
              weights=[embedding_matrix],
              trainable=False,
              input_length=sequence_length))

    for i in range(n_layers):
        if i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # first layer or hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))

    model.add(Dense(output_length, activation="softmax"))
    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


    
def load_imdb_data(num_words, sequence_length, test_size=0.25, oov_token=None):
    # read reviews
    reviews = []
    with open("data/reviews.txt") as f:
        for review in f:
            review = review.strip()
            reviews.append(review)

    labels = []
    with open("data/labels.txt") as f:
        for label in f:
            label = label.strip()
            labels.append(label)


    # tokenize the dataset corpus, delete uncommon words such as names, etc.
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(reviews)
    X = tokenizer.texts_to_sequences(reviews)
    
    X, y = np.array(X), np.array(labels)

    # pad sequences with 0's
    X = pad_sequences(X, maxlen=sequence_length)

    # convert labels to one-hot encoded
    y = to_categorical(y)

    # split data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    data = {}

    data["X_train"] = X_train
    data["X_test"]= X_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["tokenizer"] = tokenizer
    data["int2label"] =  {0: "negative", 1: "positive"}
    data["label2int"] = {"negative": 0, "positive": 1}
    
    return data


def load_20_newsgroup_data(num_words, sequence_length, test_size=0.25, oov_token=None):
    # load the 20 news groups dataset
    # shuffling the data & removing each document's header, signature blocks and quotation blocks
    dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    documents = dataset.data
    labels = dataset.target

    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(documents)
    X = tokenizer.texts_to_sequences(documents)
    
    X, y = np.array(X), np.array(labels)

    # pad sequences with 0's
    X = pad_sequences(X, maxlen=sequence_length)

    # convert labels to one-hot encoded
    y = to_categorical(y)

    # split data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    data = {}

    data["X_train"] = X_train
    data["X_test"]= X_test
    data["y_train"] = y_train
    data["y_test"] = y_test
    data["tokenizer"] = tokenizer

    data["int2label"] = { i: label for i, label in enumerate(dataset.target_names) }
    data["label2int"] = { label: i for i, label in enumerate(dataset.target_names) }
    
    return data


