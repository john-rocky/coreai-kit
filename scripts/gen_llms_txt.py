#!/usr/bin/env python3
"""Generate llms.txt — one fetch that says what this package does and where to look.

Measured reason this exists: over 14 days this repository took 662 clones from 167 unique
cloners (the SwiftPM resolution signature — people building against it), and 8 page views from
Google. Everyone who finds it arrives from GitHub or Hugging Face, which in practice means from
the model zoo. The package does not stand on its own in search, and the 28 example apps — each
one a specific, searchable task — are invisible.

Static entries for the documents; the example list is generated from each example's own H1, so
a new example is announced by existing.

    python3 scripts/gen_llms_txt.py           # regenerate
    python3 scripts/gen_llms_txt.py --check   # CI: fail if stale
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "llms.txt"
SITE = "https://john-rocky.github.io/coreai-kit"
ZOO = "https://john-rocky.github.io/coreai-model-zoo"

PREAMBLE = f"""\
# CoreAIKit

> A Swift package for running models on Apple's Core AI runtime (iOS/macOS 27) — one line per
> model across chat, vision-language, speech-to-text, text-to-speech, diarization, detection,
> depth, embeddings, OCR and forecasting. Models download from Hugging Face on first use and are
> pinned to immutable revisions. Community package, not affiliated with Apple.

```swift
import CoreAIOps
let text = try await CoreAI.transcribe(voiceMemoURL)   // speech to text, on device
let tldr = try await CoreAI.summarize(text)
```

## Start here

- [README]({SITE}/): the two layers — task ops for one-line results, model-level APIs when you
  need to pick the model, stream, or attach tools.
- [Cookbook]({SITE}/docs/COOKBOOK.html): reverse lookup. Find the task, copy the snippet; every
  one compiles with a single `import CoreAIOps`.
- [Getting started]({SITE}/docs/GETTING_STARTED.html): from an empty Xcode project to a model
  answering on device.
- [AGENTS.md]({SITE}/AGENTS.html): for coding agents — which layer to reach for, why the catalog
  is data to read rather than guess at, and the failures that only appear on a real device.
- [Stability policy]({SITE}/docs/STABILITY.html): what a version bump is allowed to change.
- [SECURITY.md]({SITE}/SECURITY.html): the trust boundary — two hosts, revision pins, and what
  is absent (no signature on the catalog, no checksum manifest, no code signing).
- [catalog.json]({SITE}/catalog.json): every model the package resolves, with its pinned
  Hugging Face revision, kind, and per-platform variants.

## Requirements

- macOS 27 beta / iOS 27 beta, Xcode 27 beta. **Real hardware only** — the Core AI framework is
  not in the iOS Simulator SDK.
- Memory, not throughput, is the shipping constraint on a phone. A 4B-class model is tight.
"""

FOOTER = f"""
## Beyond this package

- [Core AI model zoo]({ZOO}/): the models themselves — 57 ports, each with the recipe that
  produced it and a card stating what was gated and on which hardware.
- [Questions Apple's documentation doesn't answer]({ZOO}/knowledge/undocumented-answers.html):
  measured answers — the AOT threshold, the iOS-only dynamic-KV miscompile, whether a 4B fits on
  the Neural Engine, what the chunk threshold really dials.
- [The Art of Core AI](https://john-rocky.github.io/the-art-of-core-ai/): a free book built from
  the same measurements.
"""


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\*\*|`|\[|\]\([^)]*\)", "", text)).strip()


def describe(readme: Path) -> str:
    """What this example does, in one line.

    Prefer the H1 when it says something — many are written as
    "Meeting — who said what, fully on-device" and need no help. Where the H1 is just the
    directory name, the first sentence of the body is the real description, and it is
    consistently good: these READMEs open by stating the task and the model.
    """
    lines = readme.read_text().splitlines()
    name = readme.parent.name
    h1 = next((clean(l[2:]) for l in lines if l.startswith("# ")), "")
    if h1 and h1.lower().replace(" ", "") != name.lower():
        return h1

    body: list[str] = []
    for line in lines[1:]:
        if line.startswith("#") or (not line.strip() and body):
            break
        if line.strip():
            body.append(line.strip())
    sentence = clean(" ".join(body)).split(". ")[0].rstrip(".")
    return f"{name} — {sentence}" if sentence else name


def examples() -> list[tuple[str, str]]:
    """(directory, one-line description) for every buildable example."""
    return [(r.parent.name, describe(r))
            for r in sorted((REPO / "Examples").glob("*/README.md"))]


def render() -> str:
    rows = examples()
    body = [PREAMBLE, f"\n## Example apps ({len(rows)}, each buildable on its own)\n"]
    for name, title in rows:
        # An example's own README is the page; the title already says what it does.
        body.append(f"- [{name}]({SITE}/Examples/{name}/): {title}")
    body.append(FOOTER)
    return "\n".join(body).replace("\n\n\n", "\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit non-zero if llms.txt is stale")
    args = ap.parse_args()

    want = render()
    if args.check:
        have = OUT.read_text() if OUT.exists() else ""
        if have != want:
            sys.exit("llms.txt is stale — an example was added or renamed without regenerating\n"
                     "  the index. Fix with: python3 scripts/gen_llms_txt.py")
        print(f"OK: llms.txt lists {len(examples())} examples")
        return
    OUT.write_text(want)
    print(f"wrote llms.txt ({len(examples())} examples, {len(want)} bytes)")


if __name__ == "__main__":
    main()
