"""Poll the running teacher-in-the-loop RL and log progress locally.

Runs until the RL finishes normally ("saved to ckpt_rl_teacher" appears in
rl_teacher.log) or the train_rl process dies (crash/OOM). Exit code 0 = done,
1 = crashed, 2 = connect-failure (transient). Appends a timestamped record to
rl_monitor.log in this directory.
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
INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "900"))
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_monitor.log")
LOG = os.path.abspath(LOG)


def client(retries=6):
    last = None
    for a in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PW,
                      timeout=40, banner_timeout=40)
            return c
        except Exception as e:  # banner error is flaky; retry
            last = e
            time.sleep(2 * (a + 1))
    raise last


def check():
    c = client()
    try:
        cmd = (
            "cd /root/go-ai; "
            "grep -E '^iter |saved to' rl_teacher.log | tail -4; "
            "echo '---PROC---'; "
            "ls /proc/936127 >/dev/null 2>&1 && echo ALIVE || echo DEAD; "
            "echo '---GAMES---'; "
            "ls -1 selfplay_sgf_teacher/game_*.sgf 2>/dev/null | wc -l; "
            "echo '---LASTLEN---'; "
            "f=$(ls -1t selfplay_sgf_teacher/game_*.sgf 2>/dev/null | head -1); "
            "[ -n \"$f\" ] && grep -o ';' \"$f\" | wc -l || echo 0"
        )
        _i, out, err = c.exec_command(cmd, timeout=120)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        return o, e
    finally:
        c.close()


def main():
    with open(LOG, "a") as fh:
        fh.write(f"\n=== monitor start {datetime.datetime.now()} ===\n")
    while True:
        try:
            o, e = check()
        except Exception as ex:
            with open(LOG, "a") as fh:
                fh.write(f"[{datetime.datetime.now()}] CONNECT_FAIL {ex}\n")
            time.sleep(60)
            continue
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG, "a") as fh:
            fh.write(f"--- {stamp} ---\n{o}\n")
        if "saved to" in o:
            with open(LOG, "a") as fh:
                fh.write("=== RL DONE ===\n")
            print("RL DONE")
            return 0
        if "DEAD" in o and "ALIVE" not in o:
            with open(LOG, "a") as fh:
                fh.write("=== RL CRASHED (process gone) ===\n")
            print("RL CRASHED")
            return 1
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
