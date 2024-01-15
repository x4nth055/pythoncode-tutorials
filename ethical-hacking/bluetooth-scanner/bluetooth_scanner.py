# Import bluetooth from the PyBluez module.
import bluetooth

def scan_bluetooth_devices():
    try:
        # Discover Bluetooth devices with names and classes.
        discovered_devices = bluetooth.discover_devices(lookup_names=True, lookup_class=True)
        
        # Display information about the scanning process.
        print('[!] Scanning for active devices...')
        print(f"[!] Found {len(discovered_devices)} Devices\n")

        # Iterate through discovered devices and print their details.
        for addr, name, device_class in discovered_devices:
            print(f'[+] Name: {name}')
            print(f'[+] Address: {addr}')
            print(f'[+] Device Class: {device_class}\n')
    
    except Exception as e:
        # Handle and display any exceptions that occur during device discovery.
        print(f"[ERROR] An error occurred: {e}")


# Call the Bluetooth device scanning function when the script is run
scan_bluetooth_devices()
