import re

# fake ipconfig output
example_text = """
Ethernet adapter Ethernet:

   Media State . . . . . . . . . . . : Media disconnected
   Physical Address. . . . . . . . . : 88-90-E6-28-35-FA

Ethernet adapter Ethernet 2:

   Physical Address. . . . . . . . . : 04-00-4C-4F-4F-60
   Autoconfiguration IPv4 Address. . : 169.254.204.56(Preferred)

Wireless LAN adapter Local Area Connection* 2:

   Media State . . . . . . . . . . . : Media disconnected
   Physical Address. . . . . . . . . : B8-21-5E-D3-66-98

Wireless LAN adapter Wi-Fi:

   Physical Address. . . . . . . . . : A0-00-79-AA-62-74
   IPv4 Address. . . . . . . . . . . : 192.168.1.101(Preferred)
   Default Gateway . . . . . . . . . : 192.168.1.1
"""
# regex for MAC address
mac_address_regex = r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})"
# iterate over matches and extract MAC addresses
extracted_mac_addresses = [ m.group(0) for m in re.finditer(mac_address_regex, example_text) ]
print(extracted_mac_addresses)