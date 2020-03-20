from tensorflow.keras.callbacks import TensorBoard

import os

from parameters import *
from utils import create_model, load_20_newsgroup_data

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

# dataset name, IMDB movie reviews dataset
dataset_name = "20_news_group"
# get the unique model name based on hyper parameters on parameters.py
model_name = get_model_name(dataset_name)

# load the data
data = load_20_newsgroup_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token=OOV_TOKEN)

model = create_model(data["tokenizer"].word_index, units=UNITS, n_layers=N_LAYERS, 
                    cell=RNN_CELL, bidirectional=IS_BIDIRECTIONAL, embedding_size=EMBEDDING_SIZE, 
                    sequence_length=SEQUENCE_LENGTH, dropout=DROPOUT, 
                    loss=LOSS, optimizer=OPTIMIZER, output_length=data["y_train"][0].shape[0])

model.summary()

tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[tensorboard],
                    verbose=1)


model.save(os.path.join("results", model_name) + ".h5")