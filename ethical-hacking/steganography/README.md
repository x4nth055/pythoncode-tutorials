# [How to use Steganography to Hide Secret Data in Images in Python](https://www.thepythoncode.com/article/hide-secret-data-in-images-using-steganography-python)
To run this:
- `pip3 install -r requimements.txt`
-
    ```
    python steganography.py --help
    ```
    **Output**:
    ```
        usage: steganography.py [-h] [-t TEXT] [-e ENCODE] [-d DECODE]

    Steganography encoder/decoder, this Python scripts encode data within images.

    optional arguments:
    -h, --help            show this help message and exit
    -t TEXT, --text TEXT  The text data to encode into the image, this only
                            should be specified for encoding
    -e ENCODE, --encode ENCODE
                            Encode the following image
    -d DECODE, --decode DECODE
                            Decode the following image
    ```
- To encode some data to the image `image.PNG`:
    ```
    python steganography.py -e image.PNG -t "This is some secret data."
    ```
    This will write another image `image_encoded.PNG` with data encoded in it and **outputs:**
    ```
    [*] Maximum bytes to encode: 125028
    [*] Encoding data...
    [+] Saved encoded image.
    ```
- To decode the data containing the image `encoded_image.PNG`, use:
    ```
    python steganography.py -d encoded_image.PNG
    ```
    **Outputs:**
    ```
    [+] Decoding...
    [+] Decoded data: This is some secret data.
    ```

- To encode the `foo.pdf` file into the `image.PNG` image, using `steganography_advanced.py` script:
    ```
    $ python steganography_advanced.py -e image.PNG -f foo.pdf -b 1
    ```
    **Output:**
    ```
    [*] Maximum bytes to encode: 125028
    [*] Data size: 84158
    [*] Encoding data...
    [+] Saved encoded image.
    ```
- Now decoding it into a file `foo_decoded.pdf`:
    ```
    python steganography_advanced.py -d image_encoded.PNG -f foo_decoded.pdf -b 1
    [+] Decoding...
    [+] File decoded, foo_decoded.pdf is saved successfully.
    ```
