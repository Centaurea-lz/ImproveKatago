"""Poll the parallel 40b256-v2 eval (eval_b40_v2_full.log) until it finishes.

Exits 0 when the "result:" line appears, 1 if the eval process dies without a
result. Appends a timestamped record to eval_monitor.log.
"""
import datetime
import os
import sys
import time

import paramiko

HOST = "cpod-1xp9csliurw7.podtcp.compshare.cn"
PORT = 24062
USER = "root"
PW = os.environ["GO_PW"]
INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
LOG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "eval_monitor.log"))


def client(retries=6):
    last = None
    for a in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PW,
                      timeout=40, banner_timeout=40)
            return c
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise last


def check():
    c = client()
    try:
        cmd = ("cd /root/go-ai; cat eval_b40_v2_full.log 2>/dev/null; "
               "echo '---PROC---'; "
               "ps aux | grep goai.eval | grep -v grep | wc -l")
        _i, out, err = c.exec_command(cmd, timeout=120)
        return out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")
    finally:
        c.close()


def main():
    with open(LOG, "a") as fh:
        fh.write(f"\n=== eval monitor start {datetime.datetime.now()} ===\n")
    while True:
        try:
            o, e = check()
        except Exception as ex:
            with open(LOG, "a") as fh:
                fh.write(f"[{datetime.datetime.now()}] CONNECT_FAIL {ex}\n")
            time.sleep(60)
            continue
        with open(LOG, "a") as fh:
            fh.write(f"--- {datetime.datetime.now()} ---\n{o}\n")
        if "result:" in o:
            result = next((ln for ln in o.splitlines() if "result:" in ln), "")
            with open(LOG, "a") as fh:
                fh.write("=== EVAL DONE ===\n")
            print(result)
            print("EVAL DONE")
            return 0
        if o.rstrip().endswith("0"):
            with open(LOG, "a") as fh:
                fh.write("=== EVAL CRASHED (no result, proc gone) ===\n")
            print("EVAL CRASHED")
            return 1
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
