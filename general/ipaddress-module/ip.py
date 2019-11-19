import ipaddress
# initialize an IPv4 Address
ip = ipaddress.IPv4Address("192.168.1.1")

# print True if the IP address is global
print("Is global:", ip.is_global)

# print Ture if the IP address is Link-local
print("Is link-local:", ip.is_link_local)

# ip.is_reserved
# ip.is_multicast

# next ip address
print(ip + 1)

# previous ip address
print(ip - 1)

# initialize an IPv4 Network
network = ipaddress.IPv4Network("192.168.1.0/24")

# get the network mask
print("Network mask:", network.netmask)

# get the broadcast address
print("Broadcast address:", network.broadcast_address)

# print the number of IP addresses under this network
print("Number of hosts under", str(network), ":", network.num_addresses)

# iterate over all the hosts under this network
print("Hosts under", str(network), ":")
for host in network.hosts():
    print(host)

# iterate over the subnets of this network
print("Subnets:")
for subnet in network.subnets(prefixlen_diff=2):
    print(subnet)

# get the supernet of this network
print("Supernet:", network.supernet(prefixlen_diff=1))

# prefixlen_diff: An integer, the amount the prefix length of
        #   the network should be decreased by.  For example, given a
        #   /24 network and a prefixlen_diff of 3, a supernet with a
        #   /21 netmask is returned.

# tell if this network is under (or overlaps) 192.168.0.0/16
print("Overlaps 192.168.0.0/16:", network.overlaps(ipaddress.IPv4Network("192.168.0.0/16")))

