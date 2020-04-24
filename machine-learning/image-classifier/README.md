# [How to Make an Image Classifier in Python using TensorFlow and Keras](https://www.thepythoncode.com/article/image-classification-keras-python)
To run this:
- `pip3 install -r requirements.txt`
- First, you need to train the model using `python train.py`
- Edit the code in `test.py` for you optimal model weights in `results` folder ( currently does not because you need to train first ) and run:
    ```
    python test.py
    ```
    **Output:**
    ```
    10000/10000 [==============================] - 3s 331us/step
    Test accuracy: 81.17999999999999 %
    frog
    ```