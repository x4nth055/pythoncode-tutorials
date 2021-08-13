# [How to Encrypt and Decrypt PDF Files in Python](https://www.thepythoncode.com/article/encrypt-pdf-files-in-python)
To run this:
- `pip3 install -r requirements.txt`
- 
    ```
    $ python encrypt_pdf.py --help
    ```
    **Output:**
    ```
    usage: encrypt_pdf.py [-h] [-a {encrypt,decrypt}] [-l {1,2}] -p [PASSWORD] [-o OUTPUT_FILE] file

    These options are available

    positional arguments:
    file                  Input PDF file you want to encrypt

    optional arguments:
    -h, --help            show this help message and exit
    -a {encrypt,decrypt}, --action {encrypt,decrypt}
                            Choose whether to encrypt or to decrypt
    -l {1,2}, --level {1,2}
                            Choose which protection level to apply
    -p [PASSWORD], --password [PASSWORD]
                            Enter a valid password
    -o OUTPUT_FILE, --output_file OUTPUT_FILE
                            Enter a valid output file
    ```
- For instance, to encrypt `bert-paper.pdf` file and output as bert-paper-encrypted.pdf:
    ```
    $ python encrypt_pdf.py bert-paper.pdf -a encrypt -l 1 -p -o bert-paper-encrypted.pdf
    ```
- To decrypt it:
    ```
    $ python encrypt_pdf.py bert-paper-encrypted.pdf -a decrypt -l 1 -p -o bert-paper-decrypted.pdf
    ```
    This will spawn the original PDF file under the name `bert-paper-decrypted.pdf`. The password must be the same for encryption and decryption.