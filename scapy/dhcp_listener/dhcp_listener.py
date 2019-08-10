from scapy.all import *
import time

hosts = []
Ether = 1


def listen_dhcp():
    # Make sure it is DHCP with the filter options
    k = sniff(prn=print_packet, filter='udp and (port 67 or port 68)')

def print_packet(packet):
    target_mac, requested_ip, hostname, vendor_id = [None] * 4
    if packet.haslayer(Ether):
        target_mac = packet.getlayer(Ether).src
    # get the DHCP options
    dhcp_options = packet[DHCP].options
    for item in dhcp_options:
        try:
            label, value = item
        except ValueError:
            continue
        if label == 'requested_addr':
            requested_ip = value
        elif label == 'hostname':
            hostname = value.decode()
        elif label == 'vendor_class_id':
            vendor_id = value.decode()
        if target_mac and vendor_id and hostname and requested_ip and target_mac not in hosts:
            hosts.append(target_mac)
            time_now = time.strftime("[%Y-%m-%d - %H:%M:%S] ")
            print("{}: {}  -  {} / {} requested {}".format(time_now, target_mac, hostname, vendor_id, requested_ip))


if __name__ == "__main__":
    listen_dhcp()

