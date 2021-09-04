import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # only use GPU memory that we need, not allocate all the GPU memory
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)
from utils import get_model, int2label
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
import numpy as np

SEQUENCE_LENGTH = 100

# get the tokenizer
tokenizer = pickle.load(open("results/tokenizer.pickle", "rb"))

model = get_model(tokenizer, 128)
# change to the model name in results folder
model.load_weights("results/spam_classifier_0.06.h5")

def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]


while True:
    text = input("Enter the mail:")
    # convert to sequences
    print(get_predictions(text))
