# Operating system functions.
import os
# Import all functions from scapy library.
from scapy.all import *
# Import Fore from colorama for colored console output, and init for colorama initialization.
from colorama import Fore, init
# Initialize colorama
init()

# Set to store unique SSIDs.
seen_ssids = set()


# Function to set the wireless adapter to monitor mode.
def set_monitor_mode(interface):
    # Bring the interface down.
    os.system(f'ifconfig {interface} down')
    # Set the mode to monitor.
    os.system(f'iwconfig {interface} mode monitor')
    # Bring the interface back up.
    os.system(f'ifconfig {interface} up')


# Function to process Wi-Fi packets.
def process_wifi_packet(packet):
    # Check if the packet is a Probe Request, Probe Response, or Association Request.
    if packet.haslayer(Dot11ProbeReq) or packet.haslayer(Dot11ProbeResp) or packet.haslayer(Dot11AssoReq):
        # Extract SSID and BSSID from the packet.
        ssid = packet.info.decode('utf-8', errors='ignore')
        bssid = packet.addr3

        # Check if the SSID is not empty and not in the set of seen SSIDs, and if the BSSID is not the broadcast/multicast address.
        if ssid and ssid not in seen_ssids and bssid.lower() != 'ff:ff:ff:ff:ff:ff':
            # Add the SSID to the set.
            seen_ssids.add(ssid)
            # Print the identified SSID and BSSID in green.
            print(f"{Fore.GREEN}[+] SSID: {ssid} ---->  BSSID: {bssid}")


# Main function.
def main():
    # Define the wireless interface.
    wireless_interface = 'wlan0'

    # Set the wireless adapter to monitor mode.
    set_monitor_mode(wireless_interface)

    # Print a message indicating that sniffing is starting on the specified interface in magenta.
    print(f"{Fore.MAGENTA}[+] Sniffing on interface: {wireless_interface}")

    # Start sniffing Wi-Fi packets on the specified interface, calling process_wifi_packet for each packet, and disabling packet storage
    sniff(iface=wireless_interface, prn=process_wifi_packet, store=0)


# Check if the script is being run as the main program.
if __name__ == "__main__":
    # Call the main function.
    main()
