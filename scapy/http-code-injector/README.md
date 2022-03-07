# [How to Inject Code into HTTP Responses in the Network in Python](https://www.thepythoncode.com/article/injecting-code-to-html-in-a-network-scapy-python)
To run this:
- `pip3 install -r requirements.txt`
- Make sure you enabled IP forwarding, if you're using [this Python script](https://www.thepythoncode.com/code/building-arp-spoofer-using-scapy), then it'll automatically enable it.
- Start ARP Spoofing against the target using any tool such as [this Python script](https://www.thepythoncode.com/code/building-arp-spoofer-using-scapy) or arpspoof tool on Kali Linux.
- Add a new nfqueue FORWARD rule on `iptables`:
    ```bash
    $ iptables -I FORWARD -j NFQUEUE --queue-num 0
    ```

When you're done, make sure you CTRL+C the ARP spoof script, disable IP forwarding and flushing the iptables:
    ```bash
    $ iptables --flush
    ```