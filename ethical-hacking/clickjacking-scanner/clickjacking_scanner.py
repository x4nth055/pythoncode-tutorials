import requests, argparse


# Function to check if a website is vulnerable to clickjacking.
def check_clickjacking(url):
    try:
        # Add https:// schema if not present in the URL.
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url

        # Send a GET request to the URL.
        response = requests.get(url)
        headers = response.headers

        # Check for X-Frame-Options header.
        if 'X-Frame-Options' not in headers:
            return True
        
        # Get the value of X-Frame-Options and check it..
        x_frame_options = headers['X-Frame-Options'].lower()
        if x_frame_options != 'deny' and x_frame_options != 'sameorigin':
            return True
        
        return False
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while checking {url} - {e}")
        return False

# Main function to parse arguments and check the URL.
def main():
    parser = argparse.ArgumentParser(description='Clickjacking Vulnerability Scanner')
    parser.add_argument('url', type=str, help='The URL of the website to check')
    parser.add_argument('-l', '--log', action='store_true', help='Print out the response headers for analysis')
    args = parser.parse_args()

    url = args.url
    is_vulnerable = check_clickjacking(url)
    
    if is_vulnerable:
        print(f"[+] {url} may be vulnerable to clickjacking.")
    else:
        print(f"[-] {url} is not vulnerable to clickjacking.")
    
    if args.log:
        # Add https:// schema if not present in the URL for response printing.
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url

        print("\nResponse Headers:")
        response = requests.get(url)
        for header, value in response.headers.items():
            print(f"{header}: {value}")

if __name__ == '__main__':
    main()
