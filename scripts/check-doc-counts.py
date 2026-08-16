#!/usr/bin/env python3
"""Fail if a document states a catalog size that catalog.json disagrees with.

README.md and AGENTS.md both count the catalog in prose. Both said 53 while the
file held 59 — six enrollments went by without either being touched, and the
first thing an agent reads about the catalog was wrong.

A missing sentence fails too. The point is that the claim stays checked, so
rewording it out of existence should be a decision, not a silent pass.

    python3 scripts/check-doc-counts.py
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# One per file: the sentence that states the count, with the number captured.
CLAIMS = {
    "README.md": re.compile(r"All (\d+) catalog entries"),
    "AGENTS.md": re.compile(r"`catalog\.json` holds \*\*(\d+) entries\*\*"),
}


def main() -> int:
    catalog = json.loads((REPO / "catalog.json").read_text())
    models = catalog["models"] if isinstance(catalog, dict) and "models" in catalog else catalog
    actual = len(models)

    problems = []
    for name, pattern in CLAIMS.items():
        text = (REPO / name).read_text()
        found = pattern.findall(text)
        if not found:
            problems.append(f"{name}: no catalog-count sentence matched /{pattern.pattern}/")
        elif len(found) > 1:
            problems.append(f"{name}: the count is stated {len(found)} times; keep it to one")
        elif int(found[0]) != actual:
            problems.append(f"{name}: says {found[0]}, catalog.json holds {actual}")

    if problems:
        print("catalog count is out of date:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"OK: docs agree with catalog.json ({actual} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
