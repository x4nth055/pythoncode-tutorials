import sys, socket, getopt, threading, subprocess, signal, time


class NetCat:
    def __init__(self, target, port):
        self.listen = False
        self.command = False
        self.upload = False
        self.execute = ""
        self.target = target
        self.upload_destination = ""
        self.port = port
        self.running = True
        self.threads = []

    def signal_handler(self, signum, frame):
        print('\n[*] User requested an interrupt. Exiting gracefully.')
        self.running = False
        time.sleep(0.5)
        sys.exit(0)

    def run_command(self, cmd):
        cmd = cmd.rstrip()
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            output = e.output
        except Exception as e:
            output = str(e).encode()
        return output

    def handle_client(self, client_socket):
        try:
            if len(self.upload_destination):
                file_buffer = ""
                while self.running:
                    try:
                        data = client_socket.recv(1024)
                        if not data:
                            break
                        else:
                            file_buffer += data.decode('utf-8')
                    except (ConnectionResetError, BrokenPipeError) as e:
                        print(f"[!] Connection error during upload: {str(e)}")
                        break
                    except Exception as e:
                        print(f"[!] Error receiving data: {str(e)}")
                        break

                try:
                    with open(self.upload_destination, "wb") as file_descriptor:
                        file_descriptor.write(file_buffer.encode('utf-8'))
                    try:
                        client_socket.send(
                            f"Successfully saved file to {self.upload_destination}\r\n".encode('utf-8'))
                    except (BrokenPipeError, ConnectionResetError):
                        print("[!] Couldn't send success message - connection lost")
                except OSError as e:
                    print(f"[!] File operation failed: {str(e)}")
                    try:
                        client_socket.send(
                            f"Failed to save file to {self.upload_destination}\r\n".encode('utf-8'))
                    except (BrokenPipeError, ConnectionResetError):
                        print("[!] Couldn't send error message - connection lost")

            if len(self.execute) and self.running:
                try:
                    output = self.run_command(self.execute)
                    client_socket.send(output)
                except (BrokenPipeError, ConnectionResetError):
                    print("[!] Couldn't send command output - connection lost")
                except Exception as e:
                    print(f"[!] Error executing command: {str(e)}")

            if self.command:
                while self.running:
                    try:
                        # Send prompt
                        client_socket.send(b"<Target:#> ")
                        
                        # Receive command
                        cmd_buffer = b''
                        while b"\n" not in cmd_buffer and self.running:
                            try:
                                data = client_socket.recv(1024)
                                if not data:
                                    raise ConnectionResetError("No data received")
                                cmd_buffer += data
                            except socket.timeout:
                                continue
                            except (ConnectionResetError, BrokenPipeError):
                                raise
                        
                        if not self.running:
                            break

                        # Execute command and send response
                        try:
                            cmd = cmd_buffer.decode().strip()
                            if cmd.lower() in ['exit', 'quit']:
                                print("[*] User requested exit")
                                break
                                
                            output = self.run_command(cmd)
                            if output:
                                client_socket.send(output + b"\n")
                            else:
                                client_socket.send(b"Command completed without output\n")
                                
                        except (BrokenPipeError, ConnectionResetError):
                            print("[!] Connection lost while sending response")
                            break
                        except Exception as e:
                            error_msg = f"Error executing command: {str(e)}\n"
                            try:
                                client_socket.send(error_msg.encode())
                            except:
                                break

                    except ConnectionResetError:
                        print("[!] Connection reset by peer")
                        break
                    except BrokenPipeError:
                        print("[!] Broken pipe - connection lost")
                        break
                    except Exception as e:
                        print(f"[!] Error in command loop: {str(e)}")
                        break

        except Exception as e:
            print(f"[!] Exception in handle_client: {str(e)}")
        finally:
            try:
                client_socket.close()
                print("[*] Client connection closed")
            except:
                pass

    def server_loop(self):
        server = None
        try:
            if not len(self.target):
                self.target = "0.0.0.0"

            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.target, self.port))
            server.listen(5)
            
            print(f"[*] Listening on {self.target}:{self.port}")

            server.settimeout(1.0)

            while self.running:
                try:
                    client_socket, addr = server.accept()
                    print(f"[*] Accepted connection from {addr[0]}:{addr[1]}")
                    
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket,)
                    )
                    client_thread.daemon = True
                    self.threads.append(client_thread)
                    client_thread.start()

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"[!] Exception in server_loop: {str(e)}")
                    break

        except Exception as e:
            print(f"[!] Failed to create server: {str(e)}")
        finally:
            if server:
                try:
                    server.close()
                    print("[*] Server socket closed")
                except:
                    pass

            for thread in self.threads:
                try:
                    thread.join(timeout=1.0)
                except threading.ThreadError:
                    pass

    def client_sender(self, buffer):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            print(f"[*] Connecting to {self.target}:{self.port}")
            client.connect((self.target, self.port))

            if len(buffer):
                try:
                    client.send(buffer.encode('utf-8'))
                except (BrokenPipeError, ConnectionResetError):
                    print("[!] Failed to send initial buffer - connection lost")
                    return

            while self.running:
                try:
                    # Receive response from server
                    recv_len = 1
                    response = b''

                    while recv_len:
                        data = client.recv(4096)
                        recv_len = len(data)
                        response += data

                        if recv_len < 4096:
                            break

                    if response:
                        print(response.decode('utf-8'), end='')
                    
                    # Get next command
                    buffer = input()
                    if not self.running:
                        break
                    
                    if buffer.lower() in ['exit', 'quit']:
                        break

                    buffer += "\n"
                    try:
                        client.send(buffer.encode('utf-8'))
                    except (BrokenPipeError, ConnectionResetError):
                        print("\n[!] Failed to send data - connection lost")
                        break

                except ConnectionResetError:
                    print("\n[!] Connection reset by peer")
                    break
                except BrokenPipeError:
                    print("\n[!] Broken pipe - connection lost")
                    break
                except EOFError:
                    print("\n[!] EOF detected - exiting")
                    break
                except Exception as e:
                    print(f"\n[!] Exception in client loop: {str(e)}")
                    break

        except socket.error as exc:
            print("\n[!] Exception! Exiting.")
            print(f"[!] Caught exception socket.error: {exc}")
        finally:
            print("[*] Closing connection")
            try:
                client.close()
            except:
                pass

def main():
    if len(sys.argv[1:]) == 0:
        print("Custom Netcat")
        print("\nSYNOPSIS")
        print("    netcat.py [OPTIONS...]\n")
        print("OPTIONS")
        print("    -l, --listen              Start server in listening mode on specified host:port")
        print("    -e, --execute=<file>      Execute specified file upon connection establishment")
        print("    -c, --command             Initialize an interactive command shell session")
        print("    -u, --upload=<path>       Upload file to specified destination path on connection")
        print("    -t, --target=<host>       Specify target hostname or IP address")
        print("    -p, --port=<port>         Specify target port number")
        print()
        sys.exit(0)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hle:t:p:cu:",
                                   ["help", "listen", "execute", "target",
                                    "port", "command", "upload"])
        
        for o, a in opts:
            if o in ("-h", "--help"):
                main()
            elif o in ("-l", "--listen"):
                toolkit.listen = True
            elif o in ("-e", "--execute"):
                toolkit.execute = a
            elif o in ("-c", "--command"):
                toolkit.command = True
            elif o in ("-u", "--upload"):
                toolkit.upload_destination = a
            elif o in ("-t", "--target"):
                toolkit.target = a
            elif o in ("-p", "--port"):
                toolkit.port = int(a)
            else:
                assert False, "Unhandled Option"

    except getopt.GetoptError as err:
        print(str(err))
        main()

    signal.signal(signal.SIGINT, toolkit.signal_handler)
    signal.signal(signal.SIGTERM, toolkit.signal_handler)

    try:
        if not toolkit.listen and len(toolkit.target) and toolkit.port > 0:
            buffer = sys.stdin.read()
            toolkit.client_sender(buffer)

        if toolkit.listen:
            toolkit.server_loop()
    except KeyboardInterrupt:
        print("\n[*] User requested shutdown")
    except Exception as e:
        print(f"\n[!] Unexpected error: {str(e)}")
    finally:
        toolkit.running = False
        print("[*] Shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    toolkit = NetCat("", 0)
    main()