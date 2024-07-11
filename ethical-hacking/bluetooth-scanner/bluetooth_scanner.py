import bluetooth

# Major and Minor Device Class definitions based on Bluetooth specifications
MAJOR_CLASSES = {
    0: "Miscellaneous",
    1: "Computer",
    2: "Phone",
    3: "LAN/Network Access",
    4: "Audio/Video",
    5: "Peripheral",
    6: "Imaging",
    7: "Wearable",
    8: "Toy",
    9: "Health",
    10: "Uncategorized"
}

MINOR_CLASSES = {
    # Computer Major Class
    (1, 0): "Uncategorized Computer", (1, 1): "Desktop Workstation",
    (1, 2): "Server-class Computer", (1, 3): "Laptop", (1, 4): "Handheld PC/PDA",
    (1, 5): "Palm-sized PC/PDA", (1, 6): "Wearable computer",
    # Phone Major Class
    (2, 0): "Uncategorized Phone", (2, 1): "Cellular", (2, 2): "Cordless",
    (2, 3): "Smartphone", (2, 4): "Wired modem or voice gateway",
    (2, 5): "Common ISDN Access",
    # LAN/Network Access Major Class
    (3, 0): "Fully available", (3, 1): "1% to 17% utilized",
    (3, 2): "17% to 33% utilized", (3, 3): "33% to 50% utilized",
    (3, 4): "50% to 67% utilized", (3, 5): "67% to 83% utilized",
    (3, 6): "83% to 99% utilized", (3, 7): "No service available",
    # Audio/Video Major Class
    (4, 0): "Uncategorized A/V", (4, 1): "Wearable Headset", (4, 2): "Hands-free Device",
    (4, 3): "Microphone", (4, 4): "Loudspeaker", (4, 5): "Headphones", (4, 6): "Portable Audio",
    (4, 7): "Car audio", (4, 8): "Set-top box", (4, 9): "HiFi Audio Device",
    (4, 10): "VCR", (4, 11): "Video Camera", (4, 12): "Camcorder",
    (4, 13): "Video Monitor", (4, 14): "Video Display and Loudspeaker",
    (4, 15): "Video Conferencing", (4, 16): "Gaming/Toy",
    # Peripheral Major Class
    (5, 0): "Not Keyboard/Not Pointing Device", (5, 1): "Keyboard",
    (5, 2): "Pointing device", (5, 3): "Combo Keyboard/Pointing device",
    # Imaging Major Class
    (6, 0): "Display", (6, 1): "Camera", (6, 2): "Scanner", (6, 3): "Printer",
    # Wearable Major Class
    (7, 0): "Wristwatch", (7, 1): "Pager", (7, 2): "Jacket",
    (7, 3): "Helmet", (7, 4): "Glasses",
    # Toy Major Class
    (8, 0): "Robot", (8, 1): "Vehicle",
    (8, 2): "Doll / Action figure",
    (8, 3): "Controller", (8, 4): "Game",
    # Health Major Class
    (9, 0): "Undefined", (9, 1): "Blood Pressure Monitor",
    (9, 2): "Thermometer", (9, 3): "Weighing Scale",
    (9, 4): "Glucose Meter", (9, 5): "Pulse Oximeter",
    (9, 6): "Heart/Pulse Rate Monitor", (9, 7): "Health Data Display",
    (9, 8): "Step Counter", (9, 9): "Body Composition Analyzer",
    (9, 10): "Peak Flow Monitor", (9, 11): "Medication Monitor",
    (9, 12): "Knee Prosthesis", (9, 13): "Ankle Prosthesis",
    # More specific definitions can be added if needed
}

def parse_device_class(device_class):
    major = (device_class >> 8) & 0x1F # divide by 2**8 and mask with 0x1F (take the last 5 bits)
    minor = (device_class >> 2) & 0x3F # divide by 2**2 and mask with 0x3F (take the last 6 bits)
    major_class_name = MAJOR_CLASSES.get(major, "Unknown Major Class")
    minor_class_key = (major, minor)
    minor_class_name = MINOR_CLASSES.get(minor_class_key, "Unknown Minor Class")
    return major_class_name, minor_class_name


def scan_bluetooth_devices():
    try:
        discovered_devices = bluetooth.discover_devices(duration=8, lookup_names=True, lookup_class=True)
        print('[!] Scanning for Bluetooth devices...')
        print(f"[!] Found {len(discovered_devices)} Devices")
        for addr, name, device_class in discovered_devices:
            major_class, minor_class = parse_device_class(device_class)
            print(f"[+] Device Name: {name}")
            print(f"    Address: {addr}")
            print(f"    Device Class: {device_class} ({major_class}, {minor_class})")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    scan_bluetooth_devices()
