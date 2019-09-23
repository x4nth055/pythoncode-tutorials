# [How to Build a Text Generator using Keras in Python](https://www.thepythoncode.com/article/text-generation-keras-python)
To run this:
- `pip3 install -r requirements.txt`
- To use pre-trained text generator model that was trained on Alice's wonderland text book:
    ```
    python generate.py --help
    ```
    **Output:**
    ```
    usage: generate.py [-h] [-n N_CHARS] seed

    Text generator that was trained on Alice's Adventures in the Wonderland book.

    positional arguments:
    seed                  Seed text to start with, can be any english text, but
                            it's preferable you take from the book itself.

    optional arguments:
    -h, --help            show this help message and exit
    -n N_CHARS, --n-chars N_CHARS
                            Number of characters to generate, default is 200.
    ```
    Generating 200 characters with that seed:
    ```
    python generate.py "down down down there was nothing else to do so alice soon began talking again " -n 200
    ```
    **Output:**
    ```
    Generating text: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:40<00:00,  5.02it/s]
    Generated text:
    the duchess asked to the dormouse she wanted about for the world all her life i dont know what to think that it was so much sort of mine for the world a little like a stalking and was going to the mou
    ```