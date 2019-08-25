# [How to Sniff HTTP Packets in the Network using Scapy in Python](https://www.thepythoncode.com/article/sniff-http-requests-scapy-python)
to run this:
- `pip3 install -r requirements.txt`
- If you want to sniff locally ( in your PC ), you can directly run:
    ```
    python http_sniffer.py --show-raw
    ```
If you want to sniff http packets in the network, you gonna need to be man-in-the-middle using ARP spoofing, then you run this script.

You can find arp spoofer and how to use it in [here](../arp-spoofer/)