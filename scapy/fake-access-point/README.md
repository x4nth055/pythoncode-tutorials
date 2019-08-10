# [Fake Access Point Generator](https://www.thepythoncode.com/article/create-fake-access-points-scapy)
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
    python3 fake_access_point.py --help
    ```
    **Output**:
    ```
    usage: fake_access_point.py [-h] [-n N_AP] interface

    Fake Access Point Generator

    positional arguments:
    interface             The interface to send beacon frames with, must be in
                            monitor mode

    optional arguments:
    -h, --help            show this help message and exit
    -n N_AP, --access-points N_AP
                            Number of access points to be generated
    ```