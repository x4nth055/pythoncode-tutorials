import requests

url = "http://icanhazip.com"
proxy_host = "proxy.crawlera.com"
proxy_port = "8010"
proxy_auth = ":"
proxies = {
       "https": f"https://{proxy_auth}@{proxy_host}:{proxy_port}/",
       "http": f"http://{proxy_auth}@{proxy_host}:{proxy_port}/"
}

r = requests.get(url, proxies=proxies, verify=False)