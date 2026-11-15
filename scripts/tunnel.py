"""Local SSH port forward (localhost -> remote server). Run in the background.

Env: GO_HOST, GO_PORT, GO_USER, GO_PW, LOCAL_PORT (default 5000),
     REMOTE_PORT (default 5000).
"""
import os
import socket
import threading
import time

import paramiko


def main():
    host = os.environ["GO_HOST"]
    port = int(os.environ.get("GO_PORT", 22))
    user = os.environ.get("GO_USER", "root")
    pw = os.environ.get("GO_PW", "")
    local_port = int(os.environ.get("LOCAL_PORT", "5000"))
    remote_port = int(os.environ.get("REMOTE_PORT", "5000"))

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(6):
        try:
            c.connect(host, port=port, username=user, password=pw,
                      timeout=40, banner_timeout=40)
            break
        except (paramiko.SSHException, EOFError, OSError):
            time.sleep(1.5 * (attempt + 1))
    transport = c.get_transport()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", local_port))
    server.listen(64)
    print(f"TUNNEL READY: http://localhost:{local_port} -> server:{remote_port}",
          flush=True)

    def pump(src, dst):
        try:
            while True:
                d = src.recv(65536)
                if not d:
                    break
                dst.sendall(d)
        except Exception:
            pass
        finally:
            try:
                src.close()
            except Exception:
                pass
            try:
                dst.close()
            except Exception:
                pass

    while True:
        client, addr = server.accept()
        try:
            chan = transport.open_channel("direct-tcpip",
                                          ("127.0.0.1", remote_port), addr)
        except Exception:
            client.close()
            continue
        threading.Thread(target=pump, args=(client, chan), daemon=True).start()
        threading.Thread(target=pump, args=(chan, client), daemon=True).start()


if __name__ == "__main__":
    main()
