import re # stands for regular expression 
# a regular expression for validating a password
match_regex = r"^(?=.*[0-9]).{8,}$"
# a list of example passwords
passwords = ["pwd", "password", "password1"]
for pwd in passwords:
    m = re.match(match_regex, pwd)
    print(f"Password: {pwd}, validate password strength: {bool(m)}")