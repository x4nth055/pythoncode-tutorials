# [How to Crack Hashes in Python](https://thepythoncode.com/article/crack-hashes-in-python)
To run this:
- `pip install -r requirements.txt`
- Get usage: `python crack_hashes.py --help`
- Crack a SHA-256 hash using `wordlist.txt`:
    ```bash
    $ python crack_hashes.py 6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090 wordlist.txt --hash-type sha256
    ```
    **Output:**
    ```
    [*] Cracking hash 6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090 using sha256 with a list of 14344394 words.
    Cracking hash:  96%|███████████████████████████████████████████████████████████████████████████████████████████▉    | 13735317/14344394 [00:20<00:00, 664400.58it/s]
    [+] Found password: abc123
    ```