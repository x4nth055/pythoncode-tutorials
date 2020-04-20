from train import load_data, batch_size
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 classes
categories = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# load the testing set
# (_, _), (X_test, y_test) = load_data()
ds_train, ds_test, info = load_data()
# load the model with final model weights
model = load_model("results/cifar10-model-v1.h5")
# evaluation
loss, accuracy = model.evaluate(ds_test, steps=info.splits["test"].num_examples // batch_size)
print("Test accuracy:", accuracy*100, "%")

# get prediction for this image
data_sample = next(iter(ds_test))
sample_image = data_sample[0].numpy()[0]
sample_label = categories[data_sample[1].numpy()[0]]
prediction = np.argmax(model.predict(sample_image.reshape(-1, *sample_image.shape))[0])
print("Predicted label:", categories[prediction])
print("True label:", sample_label)

# show the first image
plt.axis('off')
plt.imshow(sample_image)
plt.show()