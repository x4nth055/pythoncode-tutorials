import threading
import time

def func_1():
    while True:
        print(f"[{threading.current_thread().name}] Printing this message every 2 seconds")
        time.sleep(2)

# initiate the thread with daemon set to True
daemon_thread = threading.Thread(target=func_1, name="daemon-thread", daemon=True)
# or
# daemon_thread.daemon = True
# or
# daemon_thread.setDaemon(True)
daemon_thread.start()
# sleep for 10 seconds and end the main thread
time.sleep(4)
# the main thread ends