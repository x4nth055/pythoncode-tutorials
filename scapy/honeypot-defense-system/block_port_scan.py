#!/usr/bin/env python3
"""
Honeypot Defense System
Detects port scanners using decoy ports and blocks malicious IPs
"""

from scapy.all import *
from datetime import datetime
import sys

# ==================== NETWORK INTERFACE ====================
conf.iface = "enp0s8"  # Specify your network interface

# ==================== CONFIGURATION ====================
DEFENDER_IP = "192.168.56.101"  # Change this to your Ubuntu IP

# Three-tier port system
PUBLIC_PORTS = [80]  # Open to everyone (realistic services)
HONEYPOT_PORTS = [8080, 8443, 3389, 3306]  # Decoy ports to trap attackers
PROTECTED_PORTS = [443, 53, 22, 5432]  # Hidden unless IP is allowed

ALLOWED_IPS = [
    "192.168.1.100",  # Add your Kali IP here
    "192.168.1.1",    # Add other trusted IPs
]
MAX_ATTEMPTS = 3  # Block after this many honeypot accesses (changeable)
LOG_FILE = "honeypot_logs.txt"

# ==================== GLOBALS ====================
blocked_ips = []
attempt_tracker = {}  # {IP: attempt_count}
total_scans = 0
total_blocks = 0

# ==================== HELPER FUNCTIONS ====================

def log_message(message, color_code=None):
    """Print and save log messages with timestamps"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    # Color output for terminal
    if color_code:
        print(f"\033[{color_code}m{log_entry}\033[0m")
    else:
        print(log_entry)
    
    # Save to file
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")


def is_allowed_ip(ip):
    """Check if IP is in the allowlist"""
    return ip in ALLOWED_IPS


def track_attempt(ip):
    """Track honeypot access attempts and return current count"""
    if ip not in attempt_tracker:
        attempt_tracker[ip] = 0
    attempt_tracker[ip] += 1
    return attempt_tracker[ip]


def block_ip(ip):
    """Add IP to blocklist"""
    global total_blocks
    if ip not in blocked_ips:
        blocked_ips.append(ip)
        total_blocks += 1
        log_message(f"[!] IP BLOCKED: {ip}", "91")  # Red


def create_response(packet, flags):
    """Create a TCP response packet"""
    if packet.haslayer(IP):
        response = (
            Ether(src=packet[Ether].dst, dst=packet[Ether].src) /
            IP(src=packet[IP].dst, dst=packet[IP].src) /
            TCP(
                sport=packet[TCP].dport,
                dport=packet[TCP].sport,
                flags=flags,
                seq=0,
                ack=packet[TCP].seq + 1
            )
        )
    else:  # IPv6
        response = (
            Ether(src=packet[Ether].dst, dst=packet[Ether].src) /
            IPv6(src=packet[IPv6].dst, dst=packet[IPv6].src) /
            TCP(
                sport=packet[TCP].dport,
                dport=packet[TCP].sport,
                flags=flags,
                seq=0,
                ack=packet[TCP].seq + 1
            )
        )
    return response


# ==================== MAIN PACKET HANDLER ====================

def handle_packet(packet):
    """Process incoming TCP packets with three-tier security"""
    global total_scans
    
    # Only process SYN packets (connection attempts)
    if packet[TCP].flags != "S":
        return
    
    # Extract source IP and destination port
    if packet.haslayer(IP):
        source_ip = packet[IP].src
    else:
        source_ip = packet[IPv6].src
    
    dest_port = packet[TCP].dport
    total_scans += 1
    
    # ===== CHECK IF IP IS BLOCKED FIRST =====
    if source_ip in blocked_ips:
        # Drop packet silently - no response to show as "filtered" in nmap
        log_message(f"[-] Blocked IP {source_ip} denied access to port {dest_port}", "90")
        return  # Don't send any response - this makes it appear "filtered"
    
    # ===== PUBLIC PORTS (open to everyone) =====
    if dest_port in PUBLIC_PORTS:
        # Let the real service handle it - no response needed from script
        log_message(f"[+] Public port {dest_port} accessed by {source_ip}", "94")  # Blue
        return
    
    # ===== HONEYPOT PORTS (trap for attackers) =====
    if dest_port in HONEYPOT_PORTS:
        # Always respond with SYN-ACK to appear "open"
        response = create_response(packet, "SA")
        sendp(response, verbose=False)
        
        # Check if IP is allowed
        if is_allowed_ip(source_ip):
            log_message(
                f"[+] HONEYPOT ACCESS from {source_ip}:{dest_port}\n"
                f"[!]    Status: TRUSTED IP (allowed)",
                "92"  # Green
            )
        else:
            # Track attempts for unknown IPs
            attempts = track_attempt(source_ip)
            log_message(
                f"[!] HONEYPOT ACCESS from {source_ip}:{dest_port}\n"
                f"[-]    Status: UNKNOWN IP - POTENTIAL ATTACKER\n"
                f"[!]    Strike {attempts}/{MAX_ATTEMPTS}",
                "93"  # Yellow
            )
            
            # Block after max attempts
            if attempts >= MAX_ATTEMPTS:
                block_ip(source_ip)
        return
    
    # ===== PROTECTED PORTS (only allowed IPs) =====
    if dest_port in PROTECTED_PORTS:
        if is_allowed_ip(source_ip):
            # Respond with SYN-ACK for allowed IPs
            response = create_response(packet, "SA")
            sendp(response, verbose=False)
            log_message(f"[!] Protected port {dest_port} accessed by TRUSTED IP {source_ip}", "92")
        else:
            # Drop packet silently for unknown IPs (appears filtered)
            log_message(f"[!] Protected port {dest_port} hidden from {source_ip}", "93")
        return
    
    # ===== OTHER PORTS (default behavior - drop silently) =====
    # Unknown ports are silently dropped (appear filtered)


# ==================== STARTUP & MAIN ====================

def print_banner():
    """Display startup information"""
    print("\n" + "="*60)
    print("[+]  HONEYPOT DEFENSE SYSTEM ACTIVE")
    print("="*60)
    print(f"Defending IP: {DEFENDER_IP}")
    print(f"Public Ports (open to all): {PUBLIC_PORTS}")
    print(f"Honeypot Ports (trap): {HONEYPOT_PORTS}")
    print(f"Protected Ports (allowed IPs only): {PROTECTED_PORTS}")
    print(f"Allowed IPs: {ALLOWED_IPS}")
    print(f"Block Threshold: {MAX_ATTEMPTS} attempts")
    print(f"Log File: {LOG_FILE}")
    print("="*60)
    print("Monitoring traffic... Press Ctrl+C to stop\n")


def print_summary():
    """Display statistics on exit"""
    print("\n" + "="*60)
    print("[+] SESSION SUMMARY")
    print("="*60)
    print(f"Total scans detected: {total_scans}")
    print(f"IPs blocked: {total_blocks}")
    print(f"Current blocklist: {blocked_ips if blocked_ips else 'None'}")
    print("="*60 + "\n")


def main():
    """Main execution"""
    print_banner()
    
    # Create BPF filter
    packet_filter = f"dst host {DEFENDER_IP} and tcp"
    
    try:
        # Start sniffing
        sniff(filter=packet_filter, prn=handle_packet, store=False)
    except KeyboardInterrupt:
        print("\n\n[!] Stopping honeypot defense...")
        print_summary()
        sys.exit(0)


if __name__ == "__main__":
    # Check for root privileges
    if os.geteuid() != 0:
        print("[!] This script requires root privileges. Run with: sudo python3 honeypot_defender.py")
        sys.exit(1)
    
    main()
