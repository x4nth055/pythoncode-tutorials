import requests
from stem.control import Controller
from stem import Signal

def get_tor_session():
    # initialize a requests Session
    session = requests.Session()
    # setting the proxy of both http & https to the localhost:9050 
    # (Tor service must be installed and started in your machine)
    session.proxies = {"http": "socks5://localhost:9050", "https": "socks5://localhost:9050"}
    return session

def renew_connection():
    with Controller.from_port(port=9051) as c:
        c.authenticate()
        # send NEWNYM signal to establish a new clean connection through the Tor network
        c.signal(Signal.NEWNYM)


if __name__ == "__main__":
    s = get_tor_session()
    ip = s.get("http://icanhazip.com").text
    print("IP:", ip)
    renew_connection()
    s = get_tor_session()
    ip = s.get("http://icanhazip.com").text
    print("IP:", ip)