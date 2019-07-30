import win32serviceutil
import time


class WService:

    def __init__(self, service, machine=None, verbose=False):
        self.service = service
        self.machine = machine
        self.verbose = verbose
        
    @property
    def running(self):
        return win32serviceutil.QueryServiceStatus(self.service)[1] == 4

    def start(self):
        if not self.running:
            win32serviceutil.StartService(self.service)
            time.sleep(1)
            if self.running:
                if self.verbose:
                    print(f"[+] {self.service} started successfully.")
                return True
            else:
                if self.verbose:
                    print(f"[-] Cannot start {self.service}")
                return False
        elif self.verbose:
            print(f"[!] {self.service} is already running.")
    
    def stop(self):
        if self.running:
            win32serviceutil.StopService(self.service)
            time.sleep(0.5)
            if not self.running:
                if self.verbose:
                    print(f"[+] {self.service} stopped successfully.")
                return True
            else:
                if self.verbose:
                    print(f"[-] Cannot stop {self.service}")
                return False
        elif self.verbose:
            print(f"[!] {self.service} is not running.")

    def restart(self):
        if self.running:
            win32serviceutil.RestartService(self.service)
            time.sleep(2)
            if self.running:
                if self.verbose:
                    print(f"[+] {self.service} restarted successfully.")
                return True
            else:
                if self.verbose:
                    print(f"[-] Cannot start {self.service}")
                return False
        elif self.verbose:
            print(f"[!] {self.service} is not running.")


def main(action, service):
    service = WService(service, verbose=True)
    if action == "start":
        service.start()
    elif action == "stop":
        service.stop()
    elif action == "restart":
        service.restart()

    # getattr(remoteAccessService, action, "start")()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Windows Service Handler")
    parser.add_argument("service")
    parser.add_argument("-a", "--action", help="action to do, 'start', 'stop' or 'restart'",
                        action="store", required=True, dest="action")

    given_args = parser.parse_args()

    service, action = given_args.service, given_args.action

    main(action, service)
    
