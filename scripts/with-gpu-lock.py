#!/usr/bin/env python3
"""Run a command while holding the machine-wide GPU lock.

The macOS 27 beta GPU driver can kernel-panic under parallel GPU load, so every GPU
job on a machine serializes on one lock file (macOS ships no flock(1); this wraps
fcntl). Override the lock path with COREAI_GPU_LOCK.

The command runs in its own process group. A signal to this wrapper (Ctrl-C) or
`--timeout SECONDS` ends that group and every descendant found by parentage at
that moment (xctest runs in a group of its own), SIGTERM first and SIGKILL ten
seconds later, before the lock is released, so nothing outlives the lock: on
2026-11-05 a hung `swift test` kept the lock for 26 minutes after its job was
cancelled, and every GPU job on the machine waited behind it. A cancelled
GitHub job sends this wrapper no signal (seen the same day), so a CI step
relies on `--timeout`.

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


def descendants(root: int) -> set:
    """Every pid below root by parentage right now (macOS ps has no --ppid)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    children = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, stack = set(), [root]
    while stack:
        for pid in children.get(stack.pop(), []):
            if pid not in found:
                found.add(pid)
                stack.append(pid)
    return found


def end_tree(child: subprocess.Popen, sig: int, known: set) -> None:
    """Signal the command's process group and each known descendant; a descendant
    that moved to its own group (xctest) only gets the second part."""
    signal_group(child.pid, sig)
    for pid in known:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def run_under_lock(args, timeout):
    """Run args in its own process group; end it and its descendants on a signal or after timeout."""
    child = subprocess.Popen(args, start_new_session=True)
    ended_by = []
    signal.signal(signal.SIGINT, lambda signum, _frame: ended_by.append(signum))
    signal.signal(signal.SIGTERM, lambda signum, _frame: ended_by.append(signum))
    signal.signal(signal.SIGHUP, lambda signum, _frame: ended_by.append(signum))
    deadline = None if timeout is None else time.monotonic() + timeout
    timed_out = False
    known = set()
    kill_at = None
    while True:
        try:
            status = child.wait(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if kill_at is None and (ended_by or (deadline is not None and now >= deadline)):
            if not ended_by:
                timed_out = True
                print(f"with-gpu-lock: `{' '.join(args)}` ran over {timeout:g} s; "
                      "ending it and its descendants", file=sys.stderr)
            known = descendants(child.pid)
            end_tree(child, signal.SIGTERM, known)
            kill_at = now + GRACE
        elif kill_at is not None and now >= kill_at:
            known |= descendants(child.pid)
            end_tree(child, signal.SIGKILL, known)
            kill_at = float("inf")
    if ended_by or timed_out:
        # The command is reaped; take any descendant that ignored SIGTERM with it.
        end_tree(child, signal.SIGKILL, known)
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
