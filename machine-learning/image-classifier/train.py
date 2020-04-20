from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# hyper-parameters
batch_size = 64
# 10 categories of images (CIFAR-10)
num_classes = 10
# number of training epochs
epochs = 30

def create_model(input_shape):
    """
    Constructs the model:
        - 32 Convolutional (3x3)
        - Relu
        - 32 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout

        - 64 Convolutional (3x3)
        - Relu
        - 64 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout

        - 128 Convolutional (3x3)
        - Relu
        - 128 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout
        
        - Flatten (To make a 1D vector out of convolutional layers)
        - 1024 Fully connected units
        - Relu
        - Dropout
        - 10 Fully connected units (each corresponds to a label category (cat, dog, etc.))
    """

    # building the model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # flattening the convolutions
    model.add(Flatten())
    # fully-connected layers
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    # print the summary of the model architecture
    model.summary()

    # training the model using adam optimizer
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def load_data():
    """
    This function loads CIFAR-10 dataset, and preprocess it
    """
    # Loading data using Keras 
    # loading the CIFAR-10 dataset, splitted between train and test sets
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # print("Training samples:", X_train.shape[0])
    # print("Testing samples:", X_test.shape[0])
    # print(f"Images shape: {X_train.shape[1:]}")

    # # converting image labels to binary class matrices
    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    # # convert to floats instead of int, so we can divide by 255
    # X_train = X_train.astype("float32")
    # X_test = X_test.astype("float32")
    # X_train /= 255
    # X_test /= 255
    # return (X_train, y_train), (X_test, y_test)
    # Loading data using Tensorflow Datasets
    def preprocess_image(image, label):
        # convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    # loading the CIFAR-10 dataset, splitted between train and test sets
    ds_train, info = tfds.load("cifar10", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("cifar10", split="test", as_supervised=True)
    # repeat dataset forever, shuffle, preprocess, split by batch
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info



if __name__ == "__main__":

    # load the data
    ds_train, ds_test, info = load_data()
    # (X_train, y_train), (X_test, y_test) = load_data()

    # constructs the model
    # model = create_model(input_shape=X_train.shape[1:])
    model = create_model(input_shape=info.features["image"].shape)

    # some nice callbacks
    logdir = os.path.join("logs", "cifar10-model-v1")
    tensorboard = TensorBoard(log_dir=logdir)

    # make sure results folder exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    # train
    # model.fit(X_train, y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         validation_data=(X_test, y_test),
    #         callbacks=[tensorboard, checkpoint],
    #         shuffle=True)
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1,
              steps_per_epoch=info.splits["train"].num_examples // batch_size,
              validation_steps=info.splits["test"].num_examples // batch_size,
              callbacks=[tensorboard])

    # save the model to disk
    model.save("results/cifar10-model-v1.h5")