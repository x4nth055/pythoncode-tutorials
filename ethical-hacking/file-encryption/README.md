# [How to Encrypt and Decrypt Files in Python](https://www.thepythoncode.com/article/encrypt-decrypt-files-symmetric-python)
To run this:
- `pip3 install -r requirements.txt`
- 
    ```
    python crypt --help
    ```
    **Output:**
    ```
    usage: crypt.py [-h] [-g] [-e] [-d] file

    Simple File Encryptor Script

    positional arguments:
    file                File to encrypt/decrypt

    optional arguments:
    -h, --help          show this help message and exit
    -g, --generate-key  Whether to generate a new key or use existing
    -e, --encrypt       Whether to encrypt the file, only -e or -d can be
                        specified.
    -d, --decrypt       Whether to decrypt the file, only -e or -d can be
                        specified.
    ```
- If you want to encrypt `data.csv` using a new generated key:
    ```
    python crypt.py data.csv --generate-key --encrypt
    ```
- To decrypt it (must be same key, using `--generate-key` flag with decrypt won't be able to get the original file):
    ```
    python crypt.py data.csv --decrypt
    ```
- To encrypt another file using the same key generated previously:
    ```
    python crypt.py another_file --encrypt
    ```