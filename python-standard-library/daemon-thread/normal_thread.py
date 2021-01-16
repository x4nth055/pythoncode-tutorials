import threading
import time

def func():
    while True:
        print(f"[{threading.current_thread().name}] Printing this message every 2 seconds")
        time.sleep(2)

# initiate the thread to call the above function
normal_thread = threading.Thread(target=func, name="normal_thread")
# start the thread
normal_thread.start()
# sleep for 4 seconds and end the main thread
time.sleep(4)
# the main thread ends