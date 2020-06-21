# [How to Build a SQL Injection Scanner in Python](https://www.thepythoncode.com/code/sql-injection-vulnerability-detector-in-python)
To run this:
- `pip3 install -r requirements.txt`
- Provide the URL in the command line arguments, as follows:
    ```
    python sql_injection_detector.py http://testphp.vulnweb.com/artists.php?artist=1
    ```
    **Output:**
    ```
    [!] Trying http://testphp.vulnweb.com/artists.php?artist=1"
    [+] SQL Injection vulnerability detected, link: http://testphp.vulnweb.com/artists.php?artist=1"
    ```