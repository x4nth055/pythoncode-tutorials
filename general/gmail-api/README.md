# [How to Use Gmail API in Python](https://www.thepythoncode.com/article/use-gmail-api-in-python)
To use the scripts here:
- Create your credentials file in Google API dashboard and putting it in the current directory, follow [this tutorial](https://www.thepythoncode.com/article/use-gmail-api-in-python) for detailed information.
- `pip3 install -r requirements.txt`
- Change `our_email` variable in `common.py` to your gmail address.
- To send emails, use the `send_emails.py` script:
    ```
    python send_emails.py --help
    ```
    **Output:**
    ```
    usage: send_emails.py [-h] [-f FILES [FILES ...]] destination subject body

    Email Sender using Gmail API

    positional arguments:
    destination           The destination email address
    subject               The subject of the email
    body                  The body of the email

    optional arguments:
    -h, --help            show this help message and exit
    -f FILES [FILES ...], --files FILES [FILES ...]
                            email attachments
    ```
    For example, sending to example@domain.com:
    ```
    python send_emails.py example@domain.com "This is a subject" "Body of the email" --files file1.pdf file2.txt file3.img
    ```
- To read emails, use the `read_emails.py` script. Downloading & parsing emails for Python related emails:
    ```
    python read_emails.py "python"
    ```
    This will output basic information on all matched emails and creates a folder for each email along with attachments and HTML version of the emails.
- To mark emails as **read** or **unread**, consider using `mark_emails.py`:
    ```
    python mark_emails.py --help
    ```
    **Output**:
    ```
    usage: mark_emails.py [-h] [-r] [-u] query

    Marks a set of emails as read or unread

    positional arguments:
    query         a search query that selects emails to mark

    optional arguments:
    -h, --help    show this help message and exit
    -r, --read    Whether to mark the message as read
    -u, --unread  Whether to mark the message as unread
    ```
    Marking emails from **Google Alerts** as **Read**:
    ```
    python mark_emails.py "Google Alerts" --read
    ```
    Marking emails sent from example@domain.com as **Unread**:
    ```
    python mark_emails.py "example@domain.com" -u
    ```
- To delete emails, consider using `delete_emails.py` script, e.g: for deleting emails about Bitcoin:
    ```
    python delete_emails.py "bitcoin"
    ```
- If you want the full code, consider using `tutorial.ipynb` file.
- Or if you want a all-in-one script, `gmail_api.py` is here as well!