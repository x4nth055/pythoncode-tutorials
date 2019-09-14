from train import load_data
from keras.models import load_model
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
(_, _), (X_test, y_test) = load_data()
# load the model with optimal weights
model = load_model("results/cifar10-loss-0.58-acc-0.81.h5")
# evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy*100, "%")

# get prediction for this image
sample_image = X_test[7500]
prediction = np.argmax(model.predict(sample_image.reshape(-1, *sample_image.shape))[0])
print(categories[prediction])

# show the first image
plt.axis('off')
plt.imshow(sample_image)
plt.savefig("frog.png")
plt.show()