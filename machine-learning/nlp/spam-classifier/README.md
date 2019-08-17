# [How to Build a Spam Classifier using Keras in Python](https://www.thepythoncode.com/article/build-spam-classifier-keras-python)
To run this:
- `pip3 install -r requirements.txt`
- For training, since we're using transfer learning, you first need to download, extract [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) and put to `data` folder, this is a pre trained embedding vectors that map each word to its vector, two words that have similar meanings tend to have very close vectors, and so on.
    ```
    python3 spam_classifier.py
    ```
    This will spawn tensorflow logs in `logs` folder, as well as the model and tokenizer in `results`, so `test.py` will use them.
- After the training has finished, try testing your own emails, or change the code on your needs, or whatever:
    ```
    python3 test.py
    ```