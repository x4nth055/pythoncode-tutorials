from scapy.all import *
from threading import Thread
from faker import Faker


def send_beacon(ssid, mac, infinite=True):
    dot11 = Dot11(type=0, subtype=8, addr1="ff:ff:ff:ff:ff:ff", addr2=mac, addr3=mac)
    # type=0:       management frame
    # subtype=8:    beacon frame
    # addr1:        MAC address of the receiver
    # addr2:        MAC address of the sender
    # addr3:        MAC address of the Access Point (AP)

    # beacon frame

    beacon = Dot11Beacon()
    
    # we inject the ssid name
    essid = Dot11Elt(ID="SSID", info=ssid, len=len(ssid))
    

    # stack all the layers and add a RadioTap
    frame = RadioTap()/dot11/beacon/essid

    # send the frame
    if infinite:
        sendp(frame, inter=0.1, loop=1, iface=iface, verbose=0)
    else:
        sendp(frame, iface=iface, verbose=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake Access Point Generator")
    parser.add_argument("interface", default="wlan0mon", help="The interface to send beacon frames with, must be in monitor mode")
    parser.add_argument("-n", "--access-points", type=int, dest="n_ap", help="Number of access points to be generated")
    args = parser.parse_args()
    n_ap = args.n_ap
    iface = args.interface

    # generate random SSIDs and MACs
    faker = Faker()

    ssids_macs = [ (faker.name(), faker.mac_address()) for i in range(n_ap) ]
    for ssid, mac in ssids_macs:
        Thread(target=send_beacon, args=(ssid, mac)).start()
