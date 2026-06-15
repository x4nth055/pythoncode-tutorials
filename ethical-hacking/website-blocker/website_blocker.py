#!/usr/bin/env python3
"""
Website Blocker — Block distracting websites by modifying the hosts file.

This script adds entries to your system's hosts file to redirect
specified websites to 127.0.0.1 (localhost), effectively blocking them.

Usage:
    sudo python website_blocker.py block       # Block all sites
    sudo python website_blocker.py unblock     # Unblock all sites  
    python website_blocker.py status           # Show blocked sites
"""

import sys
import platform

# ============================================================
# CONFIGURATION — edit this list to block different sites
# ============================================================

SITES_TO_BLOCK = [
    # Social media
    "www.facebook.com", "facebook.com",
    "www.twitter.com",   "twitter.com",
    "www.instagram.com", "instagram.com",
    "www.reddit.com",    "reddit.com",
    # Video / entertainment
    "www.youtube.com",   "youtube.com",
    "www.tiktok.com",    "tiktok.com",
    "www.twitch.tv",     "twitch.tv",
]

REDIRECT_IP = "127.0.0.1"

# Markers keep our entries isolated so we never touch
# other entries in the hosts file.
START_MARKER = "# >>> WEBSITE BLOCKER START >>>"
END_MARKER   = "# <<< WEBSITE BLOCKER END <<<"

# ============================================================
# Cross‑platform hosts path
# ============================================================

def get_hosts_path():
    """Return the absolute path to the hosts file for this OS."""
    system = platform.system()
    if system == "Windows":
        return r"C:\Windows\System32\drivers\etc\hosts"
    # macOS and Linux both use /etc/hosts
    return "/etc/hosts"

HOSTS_PATH = get_hosts_path()

# ============================================================
# Core operations
# ============================================================

def block_websites():
    """Write (or refresh) the blocker block into the hosts file."""
    # Read the current file
    with open(HOSTS_PATH, "r") as fh:
        content = fh.read()

    # Strip any previous block so we start fresh
    if START_MARKER in content:
        content = content.split(START_MARKER)[0].rstrip("\n") + "\n"

    # Build the block
    block_lines = [START_MARKER + "\n"]
    for site in SITES_TO_BLOCK:
        block_lines.append(f"{REDIRECT_IP}\t{site}\n")
    block_lines.append(END_MARKER + "\n")

    # Write everything back
    with open(HOSTS_PATH, "w") as fh:
        fh.write(content)
        fh.writelines(block_lines)

    unique_sites = len(SITES_TO_BLOCK) // 2
    print(f"[+] Blocked {unique_sites} websites "
          f"({len(SITES_TO_BLOCK)} URLs) → {REDIRECT_IP}")


def unblock_websites():
    """Remove the blocker block from the hosts file."""
    with open(HOSTS_PATH, "r") as fh:
        content = fh.read()

    if START_MARKER not in content:
        print("[*] No websites are currently blocked.")
        return

    # Cut out the marked section
    before = content.split(START_MARKER)[0].rstrip("\n")
    after  = content.split(END_MARKER)[-1]
    new_content = before + "\n" + after.lstrip("\n")

    with open(HOSTS_PATH, "w") as fh:
        fh.write(new_content)

    print("[+] All websites unblocked. Focus mode off.")


def show_status():
    """Print which websites are currently blocked."""
    with open(HOSTS_PATH, "r") as fh:
        content = fh.read()

    if START_MARKER not in content:
        print("[*] No websites are currently blocked.")
        return

    block = content.split(START_MARKER)[1].split(END_MARKER)[0]
    sites = [line.strip() for line in block.split("\n")
             if line.strip() and not line.strip().startswith("#")]

    print(f"[*] {len(sites)} URLs currently blocked → {REDIRECT_IP}:")
    for site in sites:
        print(f"    {site.split()[-1]}")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Website Blocker — block distracting sites via /etc/hosts\n")
        print("Usage:")
        print("  sudo python website_blocker.py block")
        print("  sudo python website_blocker.py unblock")
        print("  python website_blocker.py status")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "block":
        block_websites()
    elif command == "unblock":
        unblock_websites()
    elif command == "status":
        show_status()
    else:
        print(f"[!] Unknown command: {command}")
        print("Valid commands: block, unblock, status")
        sys.exit(1)
