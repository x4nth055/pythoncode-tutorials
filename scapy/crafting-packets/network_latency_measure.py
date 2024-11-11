server_ips = ["192.168.27.1", "192.168.17.129", "192.168.17.128"]

from scapy.all import IP, ICMP, sr1
import time

def check_latency(ip):
    packet = IP(dst=ip) / ICMP()
    start_time = time.time()
    response = sr1(packet, timeout=2, verbose=0)
    end_time = time.time()
    
    if response:
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"[+] Latency to {ip}: {latency:.2f} ms")
    else:
        print(f"[-] No response from {ip} (possible packet loss)")

for server_ip in server_ips:
    check_latency(server_ip)

   
