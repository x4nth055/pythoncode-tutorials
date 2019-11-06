import requests
from pprint import pprint

# email and password
auth = ("email@example.com", "ffffffff")

# get the HTTP Response
res = requests.get("https://secure.veesp.com/api/details", auth=auth)

# get the account details
account_details = res.json()

pprint(account_details)

# get the bought services
services = requests.get('https://secure.veesp.com/api/service', auth=auth).json()
pprint(services)

# get the upgrade options
upgrade_options = requests.get('https://secure.veesp.com/api/service/32723/upgrade', auth=auth).json()
pprint(upgrade_options)

# list all bought VMs
all_vms = requests.get("https://secure.veesp.com/api/service/32723/vms", auth=auth).json()
pprint(all_vms)

# stop a VM automatically
stopped = requests.post("https://secure.veesp.com/api/service/32723/vms/18867/stop", auth=auth).json()
print(stopped)
# {'status': True}

# start it again
started = requests.post("https://secure.veesp.com/api/service/32723/vms/18867/start", auth=auth).json()
print(started)
# {'status': True}