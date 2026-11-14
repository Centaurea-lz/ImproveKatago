"""SSH helper for the remote GPU server (paramiko).

Usage (connection info via env vars):
  GO_HOST=... GO_PORT=24062 GO_USER=root GO_PW=... python remote.py run "command"
  GO_HOST=... GO_PW=... python remote.py put <local_dir> <remote_dir>
  GO_HOST=... GO_PW=... python remote.py put_file <local_file> <remote_path>
"""
import argparse
import os
import sys
import time

import paramiko


def _client(retries=6):
    h = os.environ["GO_HOST"]
    p = int(os.environ.get("GO_PORT", 22))
    u = os.environ.get("GO_USER", "root")
    pw = os.environ.get("GO_PW", "")
    last = None
    for attempt in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(h, port=p, username=u, password=pw,
                      timeout=40, banner_timeout=40)
            return c
        except (paramiko.SSHException, EOFError, OSError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def run(cmd):
    c = _client()
    _, stdout, stderr = c.exec_command(cmd, timeout=900)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    c.close()
    return code, out, err


def put_dir(local, remote):
    c = _client()
    sftp = c.open_sftp()

    def mkdir_p(path):
        path = path.replace("\\", "/")
        try:
            sftp.stat(path)
        except IOError:
            parent = os.path.dirname(path)
            if parent and parent != path:
                mkdir_p(parent)
            sftp.mkdir(path)

    mkdir_p(remote)
    for root, dirs, files in os.walk(local):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        rel = os.path.relpath(root, local)
        target = remote if rel == "." else (remote + "/" + rel.replace("\\", "/"))
        mkdir_p(target)
        for f in files:
            if f.endswith(".pyc"):
                continue
            lp = os.path.join(root, f)
            rp = target + "/" + f
            sftp.put(lp, rp)
    sftp.close()
    c.close()


def put_file(local, remote):
    c = _client()
    sftp = c.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    c.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "put", "put_file"])
    ap.add_argument("arg1")
    ap.add_argument("arg2", nargs="?")
    a = ap.parse_args()
    if a.action == "run":
        code, out, err = run(a.arg1)
        sys.stdout.write(out)
        if err.strip():
            sys.stdout.write("--- STDERR ---\n" + err)
        sys.exit(code)
    elif a.action == "put":
        put_dir(a.arg1, a.arg2)
        print(f"uploaded {a.arg1} -> {a.arg2}")
    else:
        put_file(a.arg1, a.arg2)
        print(f"uploaded {a.arg1} -> {a.arg2}")
