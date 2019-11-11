import paramiko
import argparse

parser = argparse.ArgumentParser(description="Python script to execute BASH scripts on Linux boxes remotely.")
parser.add_argument("host", help="IP or domain of SSH Server")
parser.add_argument("-u", "--user", required=True, help="The username you want to access to.")
parser.add_argument("-p", "--password", required=True, help="The password of that user")
parser.add_argument("-b", "--bash", required=True, help="The BASH script you wanna execute")

args = parser.parse_args()
hostname = args.host
username = args.user
password = args.password
bash_script = args.bash

# initialize the SSH client
client = paramiko.SSHClient()
# add to known hosts
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(hostname=hostname, username=username, password=password)
except:
    print("[!] Cannot connect to the SSH Server")
    exit()

# read the BASH script content from the file
bash_script = open(bash_script).read()
# execute the BASH script
stdin, stdout, stderr = client.exec_command(bash_script)
# read the standard output and print it
print(stdout.read().decode())
# print errors if there are any
err = stderr.read().decode()
if err:
    print(err)
# close the connection
client.close()
