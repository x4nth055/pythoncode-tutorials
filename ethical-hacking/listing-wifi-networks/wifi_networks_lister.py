import subprocess, platform


# Get the name of the operating system.
os_name = platform.system()

# Check if the OS is Windows.
if os_name == "Windows":
    # Command to list Wi-Fi networks on Windows using netsh.
    list_networks_command = 'netsh wlan show networks'

    # Execute the command and capture the result.
    output = subprocess.check_output(list_networks_command, shell=True, text=True)

    # Print the output, all networks in range.
    print(output)

# Check if the OS is Linux.
elif os_name == "Linux":
    # Command to list Wi-Fi networks on Linux using nmcli.
    list_networks_command = "nmcli device wifi list"

    # Execute the command and capture the output.
    output = subprocess.check_output(list_networks_command, shell=True, text=True)

    # Print the output, all networks in range.
    print(output)

# Handle unsupported operating systems.
else:
    # Print a message indicating that the OS is unsupported (Not Linux or Windows).
    print("Unsupported OS")
