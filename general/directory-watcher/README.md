# [How to Create a Watchdog in Python](https://www.thepythoncode.com/article/create-a-watchdog-in-python)
To run this:
- `pip3 install -r requirements.txt`
- `python3 controller.py --help`
**Output:**
```
usage: controller.py [-h] [-d WATCH_DELAY] [-r] [-p PATTERN] [--watch-directories] path

Watchdog script for watching for files & directories' changes

positional arguments:
  path

optional arguments:
  -h, --help            show this help message and exit
  -d WATCH_DELAY, --watch-delay WATCH_DELAY
                        Watch delay, default is 1
  -r, --recursive       Whether to recursively watch for the path's children, default is False
  -p PATTERN, --pattern PATTERN
                        Pattern of files to watch, default is .txt,.trc,.log
  --watch-directories   Whether to watch directories, default is True
```
- For example, watching the path `E:\watchdog` recursively for log and text files:
    ```
    python controller.py E:\watchdog --recursive -p .txt,.log
    ```