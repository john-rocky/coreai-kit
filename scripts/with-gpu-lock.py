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
    with open(LOCK, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return subprocess.call(args)


if __name__ == "__main__":
    sys.exit(main())
