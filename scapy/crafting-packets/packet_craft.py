# Uncomment them and run according to the tutorial
#from scapy.all import IP, TCP, send, UDP

# # Step 1: Creating a simple IP packet
# packet = IP(dst="192.168.1.1")  # Setting the destination IP
# packet = IP(dst="192.168.1.1") / TCP(dport=80, sport=12345, flags="S")
# print(packet.show())  # Display packet details
# send(packet)


############
# from scapy.all import ICMP

# # Creating an ICMP Echo request packet
# icmp_packet = IP(dst="192.168.1.1") / ICMP()
# send(icmp_packet)


############
# from scapy.all import UDP

# # Creating a UDP packet
# udp_packet = IP(dst="192.168.1.1") / UDP(dport=53, sport=12345)
# send(udp_packet)



###########
# blocked_packet = IP(dst="192.168.1.1") / TCP(dport=80, flags="S")
# send(blocked_packet)

# allowed_packet = IP(dst="192.168.1.1") / UDP(dport=53)
# send(allowed_packet)

