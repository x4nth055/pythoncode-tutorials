# [How to Make a Subdomain Scanner in Python](https://www.thepythoncode.com/article/make-subdomain-scanner-python)
To run this:
- `pip3 install -r requirements.txt`
- To run the fast subdomain scanner:
    ```
    python fast_subdomain_scanner.py --help
    ```
    **Output:**
    ```
    usage: fast_subdomain_scanner.py [-h] [-l WORDLIST] [-t NUM_THREADS] domain

    Faster Subdomain Scanner using Threads

    positional arguments:
    domain                Domain to scan for subdomains without protocol (e.g
                            without 'http://' or 'https://')

    optional arguments:
    -h, --help            show this help message and exit
    -l WORDLIST, --wordlist WORDLIST
                            File that contains all subdomains to scan, line by
                            line. Default is subdomains.txt
    -t NUM_THREADS, --num-threads NUM_THREADS
                            Number of threads to use to scan the domain. Default
                            is 10
    ```
- If you want to scan hackthissite.org for subdomains using only 10 threads with a word list of 100 subdomains (`subdomains.txt`):
    ```
    python fast_subdomain_scanner.py hackthissite.org -l subdomains.txt -t 10
    ```
    After a while, it **outputs:**
    ```
    [+] Discovered subdomain: http://mail.hackthissite.org
    [+] Discovered subdomain: http://www.hackthissite.org
    [+] Discovered subdomain: http://forum.hackthissite.org
    [+] Discovered subdomain: http://admin.hackthissite.org
    [+] Discovered subdomain: http://stats.hackthissite.org
    [+] Discovered subdomain: http://forums.hackthissite.org
    ```
- For bigger subdomain wordlists, check [this repository](https://github.com/rbsec/dnscan).
