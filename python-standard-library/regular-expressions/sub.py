import re

# a basic regular expression for email matching
email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
# example text to test with
example_text = """
Subject: This is a text email!
From: John Doe <john@doe.com>
Some text here!
===============================
Subject: This is another email!
From: Abdou Rockikz <example@domain.com>
Some other text!
"""
# substitute any email found with [email protected]
print(re.sub(email_regex, "[email protected]", example_text))