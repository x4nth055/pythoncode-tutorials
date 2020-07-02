import requests
import json


# make API request and parse JSON automatically
data = requests.get("https://jsonplaceholder.typicode.com/users").json()
# save all data in a single JSON file
file_name = "user_data.json"
with open(file_name, "w") as f:
    json.dump(data, f, indent=4)
    print(file_name, "saved successfully!")

# or you can save each entry into a file
for user in data:
    # iterate over `data` list
    file_name = f"user_{user['id']}.json"
    with open(file_name, "w") as f:
        json.dump(user, f, indent=4)
        print(file_name, "saved successfully!")


# load 2nd user for instance
file_name = "user_2.json"
with open(file_name) as f:
    user_data = json.load(f)
    
print(user_data)
print("Username:", user_data["username"])