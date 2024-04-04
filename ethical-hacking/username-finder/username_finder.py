# Import necessary libraries
import requests  # For making HTTP requests
import argparse  # For parsing command line arguments
import concurrent.futures  # For concurrent execution
from collections import OrderedDict  # For maintaining order of websites
from colorama import init, Fore  # For colored terminal output
import time  # For handling time-related tasks
import random  # For generating random numbers

# Initialize colorama for colored output.
init()

# Ordered dictionary of websites to check for a given username.
WEBSITES = OrderedDict([
    ("Instagram", "https://www.instagram.com/{}"),
    ("Facebook", "https://www.facebook.com/{}"),
    ("YouTube", "https://www.youtube.com/user/{}"),
    ("Reddit", "https://www.reddit.com/user/{}"),
    ("GitHub", "https://github.com/{}"),
    ("Twitch", "https://www.twitch.tv/{}"),
    ("Pinterest", "https://www.pinterest.com/{}/"),
    ("TikTok", "https://www.tiktok.com/@{}"),
    ("Flickr", "https://www.flickr.com/photos/{}")
])

REQUEST_DELAY = 2  # Delay in seconds between requests to the same website
MAX_RETRIES = 3  # Maximum number of retries for a failed request
last_request_times = {}  # Dictionary to track the last request time for each website

def check_username(website, username):
    """
    Check if the username exists on the given website.
    Returns the full URL if the username exists, False otherwise.
    """
    url = website.format(username)  # Format the URL with the given username
    retries = 0  # Initialize retry counter

    # Retry loop
    while retries < MAX_RETRIES:
        try:
            # Implement rate limiting.
            current_time = time.time()
            if website in last_request_times and current_time - last_request_times[website] < REQUEST_DELAY:
                delay = REQUEST_DELAY - (current_time - last_request_times[website])
                time.sleep(delay)  # Sleep to maintain the request delay.

            response = requests.get(url)  # Make the HTTP request
            last_request_times[website] = time.time()  # Update the last request time.

            if response.status_code == 200:  # Check if the request was successful.
                return url
            else:
                return False
        except requests.exceptions.RequestException:
            retries += 1  # Increment retry counter on exception.
            delay = random.uniform(1, 3)  # Random delay between retries.
            time.sleep(delay)  # Sleep for the delay period.

    return False  # Return False if all retries failed.

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Check if a username exists on various websites.")
    parser.add_argument("username", help="The username to check.")
    parser.add_argument("-o", "--output", help="Path to save the results to a file.")
    args = parser.parse_args()

    username = args.username  # Username to check.
    output_file = args.output  # Output file path.

    print(f"Checking for username: {username}")

    results = OrderedDict()  # Dictionary to store results.

    # Use ThreadPoolExecutor for concurrent execution.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor.
        futures = {executor.submit(check_username, website, username): website_name for website_name, website in WEBSITES.items()}
        for future in concurrent.futures.as_completed(futures):
            website_name = futures[future]  # Get the website name.
            try:
                result = future.result()  # Get the result.
            except Exception as exc:
                print(f"{website_name} generated an exception: {exc}")
                result = False
            finally:
                results[website_name] = result  # Store the result.

    # Print the results.
    print("\nResults:")
    for website, result in results.items():
        if result:
            print(f"{Fore.GREEN}{website}: Found ({result})")
        else:
            print(f"{Fore.RED}{website}: Not Found")

    # Save results to a file if specified.
    if output_file:
        with open(output_file, "w") as f:
            for website, result in results.items():
                if result:
                    f.write(f"{website}: Found ({result})\n")
                else:
                    f.write(f"{website}: Not Found\n")
        print(f"{Fore.GREEN}\nResults saved to {output_file}")

# Call the main function
main()
