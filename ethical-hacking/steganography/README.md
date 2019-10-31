# [How to use Steganography to Hide Secret Data in Images in Python](https://www.thepythoncode.com/article/hide-secret-data-in-images-using-steganography-python)
To run this:
- `pip3 install -r requimements.txt`
- To encode some data to the imag `image.PNG` and decode it right away:
    ```
    python steganography image.PNG "This is some secret data."
    ```
    This will write another image with data encoded in it and **outputs:**
    ```
    [*] Maximum bytes to encode: 125028
    [*] Encoding data...
    [+] Decoding...
    [+] Decoded data: This is some secret data.
    ```
- You can isolate encoding and decoding processes in two different Python files, which makes more sense.
