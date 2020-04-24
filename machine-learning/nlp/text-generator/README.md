# [How to Build a Text Generator using TensorFlow and Keras in Python](https://www.thepythoncode.com/article/text-generation-keras-python)
To run this:
- `pip3 install -r requirements.txt`
- To use pre-trained text generator model that was trained on Alice's wonderland text book or Python Code:
    ```
    python generate.py --help
    ```
    This will prompt you with the choice, seed and number of characters you want!
    Here is an example on Alice's wonderland with 200 characters:
    ```
    Generated text:
    the duchess asked to the dormouse she wanted about for the world all her life i dont know what to think that it was so much sort of mine for the world a little like a stalking and was going to the mou
    ```
    Another example:
    ```
    Please choose which model you want to generate text with:
    1 - Alice's wonderland
    2 - Python Code
    2
    Enter the seed, enter q to quit, maximum 100 characters:
    import os
    import sys
    import subprocess
    import numpy as np
    q
    Enter number of characters you want to generate: 200
    Generating text: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:44<00:00,  4.68it/s]
    Generated text:
    import pandas as pd
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    config = tf.configproto(intra_op_parallelism_threads=n

    ```