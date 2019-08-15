from fbchat import Client
from fbchat.models import Message, MessageReaction

# facebook user credentials
username = "username.or.email"
password = "password"

# login
client = Client(username, password)

# get 20 users you most recently talked to
users = client.fetchThreadList()
print(users)

# get the detailed informations about these users
detailed_users = [ list(client.fetchThreadInfo(user.uid).values())[0] for user in users ]

# sort by number of messages
sorted_detailed_users = sorted(detailed_users, key=lambda u: u.message_count, reverse=True)

# print the best friend!
best_friend = sorted_detailed_users[0]

print("Best friend:", best_friend.name, "with a message count of", best_friend.message_count)

# message the best friend!
client.send(Message(
                    text=f"Congratulations {best_friend.name}, you are my best friend with {best_friend.message_count} messages!"
                    ),
            thread_id=best_friend.uid)

# get all users you talked to in messenger in your account
all_users = client.fetchAllUsers()

print("You talked with a total of", len(all_users), "users!")

# let's logout
client.logout()