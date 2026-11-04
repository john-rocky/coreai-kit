#!/usr/bin/env python3
"""Run a command while holding the machine-wide GPU lock.

The macOS 27 beta GPU driver can kernel-panic under parallel GPU load, so every GPU
job on a machine serializes on one lock file (macOS ships no flock(1); this wraps
fcntl). Override the lock path with COREAI_GPU_LOCK.

The command runs in its own process group. A signal to this wrapper (a cancelled
CI job, Ctrl-C) or `--timeout SECONDS` ends that whole group, SIGTERM first and
SIGKILL ten seconds later, before the lock is released, so no child outlives the
lock: on 2026-11-05 a hung `swift test` kept the lock for 26 minutes after its
job was cancelled, and every GPU job on the machine waited behind it.

Usage: python3 scripts/with-gpu-lock.py [--timeout SECONDS] -- <command...>
Exit status: the command's; 124 after a timeout; 128 + the signal number when a
signal ended it.
"""

import fcntl
import os
import signal
import subprocess
import sys
import time

LOCK = os.environ.get(
    "COREAI_GPU_LOCK", os.path.expanduser("~/code/coreai/_GPU_LOCK"))
GRACE = 10.0  # seconds between SIGTERM and SIGKILL


def signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except OSError:
        # ESRCH: nothing left in the group. EPERM: only zombies, or not ours.
        pass


def main() -> int:
    args = sys.argv[1:]
    timeout = None
    if args[:1] == ["--timeout"] and len(args) > 1:
        timeout = float(args[1])
        args = args[2:]
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    with open(LOCK, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        child = subprocess.Popen(args, start_new_session=True)
        ended_by = []

        def on_signal(signum, _frame):
            # Only record and forward here; the loop below does the waiting.
            ended_by.append(signum)
            signal_group(child.pid, signal.SIGTERM)

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, on_signal)
        deadline = None if timeout is None else time.monotonic() + timeout
        timed_out = False
        kill_at = None
        while True:
            try:
                status = child.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if kill_at is None and ended_by:
                kill_at = now + GRACE
            elif kill_at is None and deadline is not None and now >= deadline:
                timed_out = True
                print(f"with-gpu-lock: `{' '.join(args)}` ran over {timeout:g} s; "
                      "ending its process group", file=sys.stderr)
                signal_group(child.pid, signal.SIGTERM)
                kill_at = now + GRACE
            elif kill_at is not None and now >= kill_at:
                signal_group(child.pid, signal.SIGKILL)
                kill_at = float("inf")
        if ended_by or timed_out:
            # The command is reaped; take any grandchild that ignored SIGTERM with it.
            signal_group(child.pid, signal.SIGKILL)
        if timed_out:
            return 124
        if ended_by:
            return 128 + ended_by[0]
        return status


if __name__ == "__main__":
    sys.exit(main())
