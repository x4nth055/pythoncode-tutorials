import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.audio import MIME

# your credentials
email = "email@example.com"
password = "password"

# the sender's email
FROM = "email@example.com"
# the receiver's email
TO   = "to@example.com"
# the subject of the email (subject)
subject = "Just a subject"

# initialize the message we wanna send
msg = MIMEMultipart()
# set the sender's email
msg["From"] = FROM
# set the receiver's email
msg["To"] = TO
# set the subject
msg["Subject"] = subject
# set the body of the email
text = MIMEText("This email is sent using <b>Python</b> !", "html")
# attach this body to the email
msg.attach(text)
# initialize the SMTP server
server = smtplib.SMTP("smtp.gmail.com", 587)
# connect to the SMTP server as TLS mode (secure) and send EHLO
server.starttls()
# login to the account using the credentials
server.login(email, password)
# send the email
server.sendmail(FROM, TO, msg.as_string())
# terminate the SMTP session
server.quit()