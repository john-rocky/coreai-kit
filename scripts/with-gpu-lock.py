#!/usr/bin/env python3
"""Run a command while holding the machine-wide GPU lock.

The macOS 27 beta GPU driver can kernel-panic under parallel GPU load, so every GPU
job on a machine serializes on one lock file (macOS ships no flock(1); this wraps
fcntl). Override the lock path with COREAI_GPU_LOCK.

Usage: python3 scripts/with-gpu-lock.py -- <command...>
"""

import fcntl
import os
import subprocess
import sys
from datetime import datetime

LOCK = os.environ.get(
    "COREAI_GPU_LOCK", os.path.expanduser("~/code/coreai/_GPU_LOCK"))


def main() -> int:
    args = sys.argv[1:]
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
            return subprocess.call(args)
        finally:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()


if __name__ == "__main__":
    sys.exit(main())
