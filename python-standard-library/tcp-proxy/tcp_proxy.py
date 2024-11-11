import sys
import socket
import threading
import time
from typing import Optional, Tuple, Dict

class TcpProxy:
    def __init__(self):
        self._local_addr: str = ""
        self._local_port: int = 0
        self._remote_addr: str = ""
        self._remote_port: int = 0
        self._preload: bool = False
        self._backlog: int = 5
        self._chunk_size: int = 16
        self._timeout: int = 5
        self._buffer_size: int = 4096
        self._termination_flags: Dict[bytes, bool] = {
            b'220 ': True,
            b'331 ': True,
            b'230 ': True,
            b'530 ': True
        }
        
    def _process_data(self, stream: bytes) -> None:
        #Transform data stream for analysis
        for offset in range(0, len(stream), self._chunk_size):
            block = stream[offset:offset + self._chunk_size]
            
            # Format block representation
            bytes_view = ' '.join(f'{byte:02X}' for byte in block)
            text_view = ''.join(chr(byte) if 32 <= byte <= 126 else '.' for byte in block)
            
            # Display formatted line
            print(f"{offset:04X}   {bytes_view:<{self._chunk_size * 3}}   {text_view}")
    
    def _extract_stream(self, conn: socket.socket) -> bytes:
        #Extract data stream from connection
        accumulator = b''
        conn.settimeout(self._timeout)
        
        try:
            while True:
                fragment = conn.recv(self._buffer_size)
                if not fragment:
                    break
                    
                accumulator += fragment
                
                # Check for protocol markers
                if accumulator.endswith(b'\r\n'):
                    for flag in self._termination_flags:
                        if flag in accumulator:
                            return accumulator
                            
        except socket.timeout:
            pass
            
        return accumulator
    
    def _monitor_stream(self, direction: str, stream: bytes) -> bytes:
        # Monitor and decode stream content
        try:
            content = stream.decode('utf-8').strip()
            marker = ">>>" if direction == "in" else "<<<"
            print(f"{marker} {content}")
        except UnicodeDecodeError:
            print(f"{direction}: [binary content]")
            
        return stream
    
    def _bridge_connections(self, entry_point: socket.socket) -> None:
        #Establish and maintain connection bridge
        # Initialize exit point
        exit_point = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            exit_point.connect((self._remote_addr, self._remote_port))
            # Handle initial remote response
            if self._preload:
                remote_data = self._extract_stream(exit_point)
                if remote_data:
                    self._process_data(remote_data)
                    processed = self._monitor_stream("out", remote_data)
                    entry_point.send(processed)
            # Main interaction loop
            while True:
                # Process incoming traffic
                entry_data = self._extract_stream(entry_point)
                if entry_data:
                    print(f"\n[>] Captured {len(entry_data)} bytes incoming")
                    self._process_data(entry_data)
                    processed = self._monitor_stream("in", entry_data)
                    exit_point.send(processed)
                # Process outgoing traffic
                exit_data = self._extract_stream(exit_point)
                if exit_data:
                    print(f"\n[<] Captured {len(exit_data)} bytes outgoing")
                    self._process_data(exit_data)
                    processed = self._monitor_stream("out", exit_data)
                    entry_point.send(processed)
                # Prevent CPU saturation
                if not (entry_data or exit_data):
                    time.sleep(0.1)
        except Exception as e:
            print(f"[!] Bridge error: {str(e)}")
        finally:
            print("[*] Closing bridge")
            entry_point.close()
            exit_point.close()
    
    def orchestrate(self) -> None:
        # Orchestrate the proxy operation
        # Validate input
        if len(sys.argv[1:]) != 5:
            print("Usage: script.py [local_addr] [local_port] [remote_addr] [remote_port] [preload]")
            print("Example: script.py 127.0.0.1 8080 target.com 80 True")
            sys.exit(1)
        # Configure proxy parameters
        self._local_addr = sys.argv[1]
        self._local_port = int(sys.argv[2])
        self._remote_addr = sys.argv[3]
        self._remote_port = int(sys.argv[4])
        self._preload = "true" in sys.argv[5].lower()
        # Initialize listener
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self._local_addr, self._local_port))
        except socket.error as e:
            print(f"[!] Binding failed: {e}")
            sys.exit(1)
        listener.listen(self._backlog)
        print(f"[*] Service active on {self._local_addr}:{self._local_port}")
        # Main service loop
        while True:
            client, address = listener.accept()
            print(f"[+] Connection from {address[0]}:{address[1]}")
            bridge = threading.Thread(
                target=self._bridge_connections,
                args=(client,)
            )
            bridge.daemon = True
            bridge.start()

if __name__ == "__main__":
    bridge = TcpProxy()
    bridge.orchestrate()