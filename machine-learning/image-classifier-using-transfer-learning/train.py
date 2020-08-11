import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3 # try to use them and see which is better
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
import numpy as np

batch_size = 32
num_classes = 5
epochs = 10

IMAGE_SHAPE = (224, 224, 3)


def load_data():
    """This function downloads, extracts, loads, normalizes and one-hot encodes Flower Photos dataset"""
    # download the dataset and extract it
    data_dir = get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
    data_dir = pathlib.Path(data_dir)

    # count how many images are there
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("Number of images:", image_count)

    # get all classes for this dataset (types of flowers) excluding LICENSE file
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

    # roses = list(data_dir.glob('roses/*'))
    # 20% validation set 80% training set
    image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)

    # make the training dataset generator
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="training")
    # make the validation dataset generator
    test_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size, 
                                                        classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                                                        shuffle=True, subset="validation")

    return train_data_gen, test_data_gen, CLASS_NAMES


def create_model(input_shape):
    # load MobileNetV2
    model = MobileNetV2(input_shape=input_shape)
    # remove the last fully connected layer
    model.layers.pop()
    # freeze all the weights of the model except the last 4 layers
    for layer in model.layers[:-4]:
        layer.trainable = False
    # construct our own fully connected layer for classification
    output = Dense(num_classes, activation="softmax")
    # connect that dense layer to the model
    output = output(model.layers[-1].output)

    model = Model(inputs=model.inputs, outputs=output)

    # print the summary of the model architecture
    model.summary()

    # training the model using adam optimizer
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":

    # load the data generators
    train_generator, validation_generator, class_names = load_data()

    # constructs the model
    model = create_model(input_shape=IMAGE_SHAPE)
    # model name
    model_name = "MobileNetV2_finetune_last5"

    # some nice callbacks
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    checkpoint = ModelCheckpoint(os.path.join("results", f"{model_name}" + "-loss-{val_loss:.2f}.h5"),
                                save_best_only=True,
                                verbose=1)

    # make sure results folder exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    # count number of steps per epoch
    training_steps_per_epoch = np.ceil(train_generator.samples / batch_size)
    validation_steps_per_epoch = np.ceil(validation_generator.samples / batch_size)

    # train using the generators
    model.fit_generator(train_generator, steps_per_epoch=training_steps_per_epoch,
                        validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                        epochs=epochs, verbose=1, callbacks=[tensorboard, checkpoint])