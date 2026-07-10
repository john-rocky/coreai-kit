#!/usr/bin/env python3
"""Pin every catalog.json entry to the current Hub revision of its repo.

Each entry carries a "revision" field holding a Hugging Face commit hash. CoreAIKit
downloads bundles from exactly that revision, so a later push to a model repo cannot
change what apps receive until the pin is deliberately bumped by re-running this
script and committing the catalog change.

Usage:
  python3 scripts/pin-catalog.py            # re-pin all entries to each repo's current main
  python3 scripts/pin-catalog.py --check    # verify every entry is pinned; report drift
"""

import json
import sys
import urllib.request
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / "catalog.json"


def head_sha(repo: str) -> str:
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["sha"]


def with_revision(entry: dict, sha: str) -> dict:
    """Rebuild the entry with "revision" placed right after "repo" (stable diffs)."""
    rebuilt = {}
    for key, value in entry.items():
        if key == "revision":
            continue
        rebuilt[key] = value
        if key == "repo":
            rebuilt["revision"] = sha
    return rebuilt


def main() -> int:
    check_only = "--check" in sys.argv
    catalog = json.loads(CATALOG.read_text())
    missing, drifted, failed = [], [], []

    for i, entry in enumerate(catalog["models"]):
        repo, pin = entry["repo"], entry.get("revision")
        try:
            sha = head_sha(repo)
        except Exception as err:  # 404 (unpublished repo), network, ...
            failed.append((entry["id"], repo, err))
            continue
        if pin is None:
            missing.append(entry["id"])
        elif pin != sha:
            drifted.append((entry["id"], pin[:12], sha[:12]))
        if not check_only:
            catalog["models"][i] = with_revision(entry, sha)
        print(f"  {entry['id']}: {sha[:12]}" + (" (was unpinned)" if pin is None else ""))

    for model_id, repo, err in failed:
        print(f"WARNING: could not resolve {repo} for '{model_id}': {err}", file=sys.stderr)

    if check_only:
        for model_id, old, new in drifted:
            print(f"drift: {model_id} pinned {old}, repo main is now {new}")
        if missing:
            print(f"FAIL: unpinned entries: {', '.join(missing)}", file=sys.stderr)
            return 1
        print(f"OK: all {len(catalog['models'])} entries pinned "
              f"({len(drifted)} trail their repo's main — bump deliberately).")
        return 0

    CATALOG.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    print(f"Pinned {len(catalog['models']) - len(failed)}/{len(catalog['models'])} entries.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
