#!/usr/bin/env python3
"""Measure what each catalog entry actually downloads, against what it claims.

    python3 scripts/measure-catalog-sizes.py            # the table
    python3 scripts/measure-catalog-sizes.py --write    # correct catalog.json

`sizeMB` is the number an adopter is shown before committing to a download, and the number
`capability()` will answer with. It is hand-maintained, so it drifts — and it had, badly:
several iOS entries looked like the macOS figure doubled rather than measured, which is what
an AOT bundle does for *some* models and not others.

**It measures one thing precisely and refuses to guess the rest**, which is the only honest
shape for this. What it measures is the bytes under the variant's path. For the common
`macos/` | `ios/` layout that is the whole download — bundle plus tokenizer — and the number
can be corrected mechanically.

Where it stops: an entry whose path names a single file, or a repo with no platform
directories, has sibling subtrees that only the *loader* knows to fetch — a host glue
directory, voice packs, a paired vision tower, the PLE tables beside a Gemma 4 decoder. Those
entries deliberately declare the total, this cannot see it, and `--write` leaves them alone
rather than silently under-reporting a multi-gigabyte model as its decoder.

Two ways this was wrong before it was right, both worth knowing before trusting a number:

  * Summing "everything that is not the other platform" reports 9.9 GB for a 1.3 GB model,
    because several repos publish six quantisations side by side and only one is resolved.
  * Treating the other platform's artifact as a sibling doubles every entry whose path names
    a file — which is exactly the shape of the errors this was written to find.

Reads file metadata only; no weights are downloaded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("needs huggingface_hub: pip install huggingface_hub")

ROOT = Path(__file__).resolve().parent.parent
PLATFORMS = ("macos", "ios")

# Never downloaded by a loader: repo furniture and the card's media.
IGNORED_SUFFIXES = (".md", ".gif", ".jpg", ".jpeg", ".png", ".gitattributes", ".txt")


def is_asset(name: str) -> bool:
    return not name.lower().endswith(IGNORED_SUFFIXES)


def measure(api: HfApi, entry: dict, platform: str) -> tuple[int, list[str]] | None:
    """(bytes, notes) that a first run on `platform` pulls, or None if not published there."""
    variant = (entry.get("variants") or {}).get(platform)
    if not variant:
        return None
    try:
        info = api.repo_info(entry["repo"], revision=entry.get("revision"),
                            files_metadata=True)
    except Exception as error:  # noqa: BLE001 — the report is the point, not the traceback
        return (0, [f"unreadable: {error}"])

    path = (variant.get("path") or "").strip("/")
    files = [(f.rfilename, f.size or 0) for f in info.siblings if is_asset(f.rfilename)]
    notes: list[str] = []

    # Everything under the variant's path. For the common `macos/` | `ios/` layout that is the
    # bundle *and* its tokenizer, which is what the entry is supposed to declare.
    #
    # It is deliberately NOT "everything in the repo that is not the other platform". Several
    # repos carry alternative artifacts side by side — qwen3.5-0.8B publishes six quantisations
    # under `gpu-pipelined/` and two more under `ios-gpu/` — and summing those reports 9.9 GB
    # for a 1.3 GB model. (Written that way first; the number was so wrong it was obvious.)
    total = sum(size for name, size in files if not path or name.startswith(path + "/")
                or name == path)

    # A path naming one file, or a repo whose bundle sits at the root, means the loader also
    # resolves siblings this cannot see — a tokenizer, host glue, voice packs, a vision tower.
    # Those totals need the loader's knowledge and are reported rather than guessed at.
    if not path or re.search(r"\.(aimodel|aimodelc)$", path):
        # The other platform's artifact is not a sibling — it is the thing this variant
        # exists instead of.
        others = {(entry["variants"].get(p) or {}).get("path", "") for p in PLATFORMS
                  if p != platform}
        siblings = sorted({n.split("/")[0] for n, _ in files
                           if not n.startswith(path) and "/" in n
                           and not any(o and n.startswith(o) for o in others)})
        if siblings:
            notes.append("loader also pulls: " + ", ".join(siblings[:4]))
    return total, notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="correct catalog.json in place")
    parser.add_argument("--only", help="one catalog id")
    args = parser.parse_args()

    catalog_path = ROOT / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    api = HfApi()

    print(f"{'model':<32}{'platform':<8}{'declared':>10}{'measured':>10}{'delta':>9}")
    changed = 0
    for entry in catalog["models"]:
        if args.only and entry["id"] != args.only:
            continue
        for platform in PLATFORMS:
            result = measure(api, entry, platform)
            if result is None:
                continue
            measured_bytes, notes = result
            measured = round(measured_bytes / 1_000_000)
            declared = (entry["variants"][platform] or {}).get("sizeMB")
            if measured == 0:
                print(f"{entry['id']:<32}{platform:<8}{str(declared):>10}{'?':>10}"
                      f"{'':>9}  {'; '.join(notes)}")
                continue
            ratio = measured / declared if declared else 0
            flag = "" if declared and 0.9 <= ratio <= 1.1 else "  <-"
            print(f"{entry['id']:<32}{platform:<8}{str(declared):>10}{measured:>10}"
                  f"{(measured - (declared or 0)):>+9}{flag}"
                  + (("  " + "; ".join(notes)) if notes else ""))
            if args.write and declared != measured and not notes:
                entry["variants"][platform]["sizeMB"] = measured
                changed += 1

    if args.write and changed:
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
        print(f"\nwrote {changed} corrected sizes to {catalog_path.name}")
        print("`ModelCatalog.builtinLiteral` in Sources/CoreAIKitCore/ModelCatalog.swift "
              "carries the same numbers — update it too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
