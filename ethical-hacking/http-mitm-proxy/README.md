# [How to Make an HTTP Proxy in Python](https://www.thepythoncode.com/article/writing-http-proxy-in-python-with-mitmproxy)
To run this:
- Install [mitmproxy](https://mitmproxy.org/).
- Run the following command:
    ```
    $ mitmproxy --ignore '^(?!duckduckgo\.com)' -s proxy.py
    ```
- Test your proxy via configuring your browser or tools such as iptables (check [the tutorial](https://www.thepythoncode.com/article/writing-http-proxy-in-python-with-mitmproxy) for more info), or you can test it out with `curl`:
    ```
    $ curl -x http://127.0.0.1:8080/ -k https://duckduckgo.com/
    ```