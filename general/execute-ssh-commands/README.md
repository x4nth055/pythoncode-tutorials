# [How to Execute BASH Commands in a Remote Machine in Python](https://www.thepythoncode.com/article/executing-bash-commands-remotely-in-python)
To run this:
- `pip3 install -r requirements.txt`
- To execute certain commands, edit `execute_commands.py` on your needs and then execute.
- To execute an entire BASH script (.sh) named `script.sh` for instance on `192.168.1.101` with `test` as username and `abc123` as password:
    ```
    python execute_bash.py 192.168.1.101 -u root -p inventedpassword123 -b script.sh
    ```