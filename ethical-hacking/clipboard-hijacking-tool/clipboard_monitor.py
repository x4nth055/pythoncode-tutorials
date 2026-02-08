import win32gui
import win32api
import ctypes
from win32clipboard import GetClipboardOwner
from win32process import GetWindowThreadProcessId
from psutil import Process
import winsound
import sys
import signal

def handle_clipboard_event(window_handle, message, w_param, l_param):
    if message == 0x031D:  # WM_CLIPBOARDUPDATE
        try:
            clipboard_owner_window = GetClipboardOwner()
            process_id = GetWindowThreadProcessId(clipboard_owner_window)[1]
            process = Process(process_id)
            process_name = process.name()
            
            # Successfully identified the process - no beep
            print("Clipboard modified by %s" % process_name)
            
        except Exception:
            # Could not identify the process - BEEP!
            print("Clipboard modified by unknown process")
            winsound.Beep(1000, 300)

    return 0


def create_listener_window():
    window_class = win32gui.WNDCLASS()
    window_class.lpfnWndProc = handle_clipboard_event
    window_class.lpszClassName = 'clipboardListener'
    window_class.hInstance = win32api.GetModuleHandle(None)

    class_atom = win32gui.RegisterClass(window_class)

    return win32gui.CreateWindow(
        class_atom,
        'clipboardListener',
        0,
        0, 0, 0, 0,
        0, 0,
        window_class.hInstance,
        None
    )


def signal_handler(sig, frame):
    print("\n[+] Exiting...")
    sys.exit(0)


def start_clipboard_monitor():
    print("[+] Clipboard listener started")
    print("[+] Press Ctrl+C to exit\n")
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    listener_window = create_listener_window()
    ctypes.windll.user32.AddClipboardFormatListener(listener_window)
    
    # Pump messages but check for exit condition
    try:
        while True:
            # Process messages with a timeout to allow checking for exit
            if win32gui.PumpWaitingMessages() != 0:
                break
            win32api.Sleep(100)  # Sleep a bit to prevent high CPU usage
    except KeyboardInterrupt:
        print("\n[+] Exiting...")
    finally:
        # Clean up - remove clipboard listener
        ctypes.windll.user32.RemoveClipboardFormatListener(listener_window)


if __name__ == "__main__":
    start_clipboard_monitor()