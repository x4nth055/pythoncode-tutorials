# Import the neccasary modules.
import sys
from scapy.all import sr, IP, ICMP
from faker import Faker
from colorama import Fore, init

# Initialize colorama for colored console output.
init()
# Create a Faker object for generating fake data.
fake = Faker()

# Function to generate a fake IPv4 address.
def generate_fake_ip():
    return fake.ipv4()

# Function to craft and send an ICMP packet.
def craft_and_send_packet(source_ip, destination_ip):
    # Craft an ICMP packet with the specified source and destination IP.
    packet = IP(src=source_ip, dst=destination_ip) / ICMP()
    # Send and receive the packet with a timeout.
    answers, _ = sr(packet, verbose=0, timeout=5)
    return answers

# Function to display a summary of the sent and received packets.
def display_packet_summary(sent, received):
    print(f"{Fore.GREEN}[+] Sent Packet: {sent.summary()}\n")
    print(f"{Fore.MAGENTA}[+] Response: {received.summary()}")

# Check if the correct number of command-line arguments is provided.
if len(sys.argv) != 2:
    print(f"{Fore.RED}[-] Error! {Fore.GREEN} Please run as: {sys.argv[0]} <dst_ip>")
    sys.exit(1)

# Retrieve the destination IP from the command-line arguments.
destination_ip = sys.argv[1]
# Generate a fake source IP.
source_ip = generate_fake_ip()
# Craft and send the packet, and receive the response.
answers = craft_and_send_packet(source_ip, destination_ip)
# Display the packet summary for each sent and received pair.
for sent, received in answers:
    display_packet_summary(sent, received)
