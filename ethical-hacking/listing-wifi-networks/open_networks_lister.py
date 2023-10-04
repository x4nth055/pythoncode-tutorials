import subprocess, platform, re
from colorama import init, Fore

init()


def list_open_networks():
    # Get the name of the operating system.
    os_name = platform.system()

    # Check if the OS is Windows.
    if os_name == "Windows":
        # Command to list Wi-Fi networks on Windows.
        list_networks_command = 'netsh wlan show networks'
        try:
            # Execute the command and capture the output.
            output = subprocess.check_output(list_networks_command, shell=True, text=True)
            networks = []

            # Parse the output to find open Wi-Fi networks.
            for line in output.splitlines():
                if "SSID" in line:
                    # Extract the SSID (Wi-Fi network name).
                    ssid = line.split(":")[1].strip()
                elif "Authentication" in line and "Open" in line:
                    # Check if the Wi-Fi network has open authentication.
                    networks.append(ssid)

            # Check if any open networks were found.
            if len(networks) > 0:
                # Print a message for open networks with colored output.
                print(f'{Fore.LIGHTMAGENTA_EX}[+] Open Wifi networks in range: \n')
                for each_network in networks:
                    print(f"{Fore.GREEN}[+] {each_network}")
            else:
                # Print a message if no open networks were found.
                print(f"{Fore.RED}[-] No open wifi networks in range")

        except subprocess.CalledProcessError as e:
            # Handle any errors that occur during the execution of the command.
            print(f"{Fore.RED}Error: {e}")
            # Return an empty list to indicate that no networks were found.
            return []

    elif os_name == "Linux":
        try:
            # Run nmcli to list available Wi-Fi networks.
            result = subprocess.run(["nmcli", "--fields", "SECURITY,SSID", "device", "wifi", "list"],
                                    stdout=subprocess.PIPE,
                                    text=True, check=True)

            # Access the captured stdout.
            output = result.stdout.strip()

            # Define a regex pattern to capture SSID and Security.
            pattern = re.compile(r'^(?P<security>[^\s]+)\s+(?P<ssid>.+)$', re.MULTILINE)

            # Find all matches in the output.
            matches = pattern.finditer(output)

            # Skip the first match, which is the header.
            next(matches, None)
            print(f"{Fore.LIGHTMAGENTA_EX}[+] Open Wifi networks in range: \n")
            # Loop through all matches (results)
            for match in matches:
                security = match.group('security')
                ssid = match.group('ssid')
                full_match = f"{Fore.GREEN}[+] SSID: {ssid} -------> Security: {security}"
                # Check if the indicator of an open network in our Full match (result).
                if "Security: --" in full_match:
                    print(f"{Fore.GREEN}[+] {ssid}")
                else:
                    print(f"{Fore.RED}[-] No open Wifi networks in range.")

        except subprocess.CalledProcessError as e:
            print(f"Error running nmcli: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        print(f"{Fore.RED}Unsupported operating system.")
        return []


# Call the function.
list_open_networks()


