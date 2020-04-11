import ftplib

# FTP server credentials
FTP_HOST = "ftp.dlptest.com"
FTP_USER = "dlpuser@dlptest.com"
FTP_PASS = "SzMf7rTE4pCrf9dV286GuNe4N"

# connect to the FTP server
ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
# force UTF-8 encoding
ftp.encoding = "utf-8"
# local file name you want to upload
filename = "some_file.txt"
with open(filename, "rb") as file:
    # use FTP's STOR command to upload the file
    ftp.storbinary(f"STOR {filename}", file)
# list current files & directories
ftp.dir()
# quit and close the connection
ftp.quit()