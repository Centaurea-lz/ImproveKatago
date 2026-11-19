"""Watch the KataGo low-visit ladder (run_ladder.sh) until it completes.

Exits 0 when "LADDER COMPLETE" appears in ladder.log; on completion it gathers
the "result:" line from each opponent's log and prints them all.
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
INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "600"))
LOG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "ladder_monitor.log"))


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
        cmd = ("cd /root/go-ai; "
               "tail -2 ladder2.log 2>/dev/null; "
               "for N in 1 5 10 50; do "
               "  echo \"=== katago ${N}visit ===\"; "
               "  grep -E '^result:' eval_b40_v2_katago_${N}visit.log 2>/dev/null; "
               "done; "
               "echo '---PROC---'; "
               "ps aux | grep -E 'run_ladder|goai.eval' | grep -v grep | wc -l")
        _i, out, err = c.exec_command(cmd, timeout=120)
        return out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")
    finally:
        c.close()


def main():
    with open(LOG, "a") as fh:
        fh.write(f"\n=== ladder monitor start {datetime.datetime.now()} ===\n")
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
        if "LADDER COMPLETE" in o:
            results = [ln for ln in o.splitlines() if "result:" in ln]
            with open(LOG, "a") as fh:
                fh.write("=== LADDER DONE ===\n")
            print("\n".join(results))
            print("LADDER DONE")
            return 0
        if o.rstrip().endswith("0"):
            with open(LOG, "a") as fh:
                fh.write("=== LADDER STOPPED (no complete, procs gone) ===\n")
            print("LADDER STOPPED")
            return 1
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
