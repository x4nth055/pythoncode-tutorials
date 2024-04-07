import winreg  # Import registry module.

def val2addr(val):  # Convert value to address format.
    addr = ''  # Initialize address.
    try:
        for ch in val:  # Loop through value characters.
            addr += '%02x ' % ch  # Convert each character to hexadecimal.
        addr = addr.strip(' ').replace(' ', ':')[0:17]  # Format address.
    except:
        return "N/A" # Return N/A if error occurs.
    return addr  # Return formatted address.


def printNets():  # Print network information.
    net = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\NetworkList\Signatures\Unmanaged"  # Registry key for network info.
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, net)  # Open registry key.
    print('\n[*] Networks You have Joined:')  # Print header.
    for i in range(100):  # Loop through possible network keys.
        try:
            guid = winreg.EnumKey(key, i)  # Get network key.
            netKey = winreg.OpenKey(key, guid)  # Open network key.
            try:
                n, addr, t = winreg.EnumValue(netKey, 5)  # Get MAC address.
                n, name, t = winreg.EnumValue(netKey, 4)  # Get network name.
                if addr:
                    macAddr = val2addr(addr)  # Convert MAC address.
                else:
                    macAddr = 'N/A'
                netName = str(name)  # Convert network name to string.
                print(f'[+] {netName} ----> {macAddr}')  # Print network info.
            except WindowsError:  # Handle errors.
                pass  # Continue loop.
            winreg.CloseKey(netKey)  # Close network key.
        except WindowsError:  # Handle errors.
            break  # Exit loop.
    winreg.CloseKey(key)  # Close registry key.

 
printNets()  # Call printNets function.
