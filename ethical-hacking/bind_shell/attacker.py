import socket,os

SERVER_HOST = input("Enter host: ")
SERVER_PORT = int(input("Enter port: "))
BUFFER_SIZE = 1024 * 128 # 128KB max size of messages, feel free to increase
# separator string for sending 2 messages in one go
SEPARATOR = "<sep>"

# create the socket object
s = socket.socket()
# connect to the server
s.connect((SERVER_HOST, SERVER_PORT))
if(s.recv(BUFFER_SIZE).decode()) == "auth": 
    password = input("Enter password to connect: ")
    s.send(password.encode())
    res = s.recv(BUFFER_SIZE).decode()
    if res == "denied":
        print("Authentication denied")
        exit()
    else:
        cwd = res  
        print("connected :-)")
        print(f"[+] pwd : {cwd}\n\n")
        while True:
            cmd = input(f"{cwd} $ ")
            s.send(cmd.encode())
            output = s.recv(BUFFER_SIZE).decode()
            cwd = output.split(SEPARATOR)[0]
            if output.split(SEPARATOR)[1] == "exiting...":
                print("exiting...")
                break
            print(output.split(SEPARATOR)[1])
s.close()
