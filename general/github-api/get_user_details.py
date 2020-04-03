import requests
from pprint import pprint

# github username
username = "x4nth055"
# url to request
url = f"https://api.github.com/users/{username}"
# make the request and return the json
user_data = requests.get(url).json()
# pretty print JSON data
pprint(user_data)
# get name
name = user_data["name"]
# get blog url if there is
blog = user_data["blog"]
# extract location
location = user_data["location"]
# get email address that is publicly available
email = user_data["email"]
# number of public repositories
public_repos = user_data["public_repos"]
# get number of public gists
public_gists = user_data["public_gists"]
# number of followers
followers = user_data["followers"]
# number of following
following = user_data["following"]
# date of account creation
date_created = user_data["created_at"]
# date of account last update
date_updated = user_data["updated_at"]
# urls
followers_url = user_data["followers_url"]
following_url = user_data["following_url"]

# print all
print("User:", username)
print("Name:", name)
print("Blog:", blog)
print("Location:", location)
print("Email:", email)
print("Total Public repositories:", public_repos)
print("Total Public Gists:", public_gists)
print("Total followers:", followers)
print("Total following:", following)
print("Date Created:", date_created)
print("Date Updated:", date_updated)

