#!/usr/bin/env python3
"""Drop "beta" from the OS requirement, on the day macOS/iOS 27 ships.

Three sentences across three docs files tell a reader they need a beta OS
(the README already reads GA — its beta wording was dropped with the 2026-08-31
quickstart rework). On GA day every one of them becomes wrong, and a reader who
believes them concludes the package is not for their machine. This is the edit,
written ahead of time so that day costs a command instead of a search.

    scripts/ga-wording.py --check    # every pre-GA sentence is still present
    scripts/ga-wording.py --apply    # make the edit

`--check` runs in CI. Its job is to fail *before* GA, not after: if someone
rewords one of these lines the exact-match replacement here goes stale, and a
prepared edit that silently no longer applies is worse than no prepared edit —
on the morning it is needed nobody has time to find out why it did nothing.

After GA, run --apply, commit, and delete this script along with its CI step.
The full GA-day runbook (OS + Xcode pin flip, gates, the 0.4.x tag) is
docs/GA_DAY.md; this script is its step 3.
"""

import argparse
import pathlib
import sys

# (file, before, after). Exact strings, so a reworded line is caught rather
# than half-edited.
EDITS = [
    (
        "docs/GETTING_STARTED.md",
        "- macOS 27 beta or iOS 27 beta (real device — the CoreAI framework is not in the iOS\n"
        "  Simulator SDK), Xcode 27 beta",
        "- macOS 27 or iOS 27 (real device — the CoreAI framework is not in the iOS\n"
        "  Simulator SDK), Xcode 27",
    ),
    (
        "docs/COOKBOOK.md",
        "(macOS 27 beta / iOS 27 beta) and downloads its model from the Hugging Face Hub on",
        "(macOS 27 / iOS 27) and downloads its model from the Hugging Face Hub on",
    ),
    (
        "docs/STABILITY.md",
        "- **CI** — build + tests + pin check on every push (macOS 27 beta, self-hosted).",
        "- **CI** — build + tests + pin check on every push (macOS 27, self-hosted).",
    ),
]

root = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="verify every edit still applies")
    group.add_argument("--apply", action="store_true", help="make the edit")
    args = ap.parse_args()

    stale = []
    done = []
    for name, before, after in EDITS:
        text = (root / name).read_text()
        if before in text:
            done.append((name, before, after))
        elif after in text:
            stale.append(f"{name}: already reads as GA — this edit has been applied")
        else:
            stale.append(f"{name}: neither the pre-GA nor the GA sentence is present:\n    {before!r}")

    if stale:
        print("GA wording edit is out of date:", file=sys.stderr)
        for line in stale:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nFix scripts/ga-wording.py to match what the docs now say, or drop the entry.",
            file=sys.stderr,
        )
        return 1

    if args.check:
        print(f"OK: {len(done)} GA wording edits still apply cleanly.")
        return 0

    for name, before, after in done:
        path = root / name
        path.write_text(path.read_text().replace(before, after))
        print(f"{name}: updated")
    print("\nRemaining by hand: nothing. Delete this script and its CI step with the commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
