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
from datetime import datetime

LOCK = os.environ.get(
    "COREAI_GPU_LOCK", os.path.expanduser("~/code/coreai/_GPU_LOCK"))
GRACE = 10.0  # seconds between SIGTERM and SIGKILL


def signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except OSError:
        # ESRCH: nothing left in the group. EPERM: only zombies, or not ours.
        pass


def run_under_lock(args, timeout):
    """Run args in its own process group; end the group on a signal or after timeout."""
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
    # Open without truncating. The file's content is the current holder's label (a measurement
    # window writes `<label> timing pid <n> since <time>` and other tools read it to stay out of the
    # window); opening with "w" blanked that label while the holder still held the flock, which
    # switched those guards off for the rest of the holder's window (seen 2026-10-14, two CI runs on
    # a self-hosted runner that shares the machine). Write our own label only once the lock is ours,
    # and leave the file empty when we release it.
    with open(LOCK, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        since = datetime.now().astimezone().isoformat(timespec="seconds")
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"coreai-kit with-gpu-lock pid {os.getpid()} since {since} ({' '.join(args)[:80]})\n")
        lock_file.flush()
        try:
            return run_under_lock(args, timeout)
        finally:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()


if __name__ == "__main__":
    sys.exit(main())
