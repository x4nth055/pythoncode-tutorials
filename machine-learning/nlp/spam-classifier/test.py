import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )
from utils import get_model, int2label, label2int
from keras.preprocessing.sequence import pad_sequences

import pickle
import numpy as np

SEQUENCE_LENGTH = 100

# get the tokenizer
tokenizer = pickle.load(open("results/tokenizer.pickle", "rb"))

model = get_model(tokenizer, 128)
model.load_weights("results/spam_classifier_0.05")

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
