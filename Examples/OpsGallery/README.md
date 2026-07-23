# OpsGallery

Every task op as a card. The grid renders straight from `CoreAI.Op.allCases` — each
card shows the op's `summary` ("Image → labeled bounding boxes") and `defaultModelID`
from the kit's own registry, so a new op in the package appears here without app
changes. Tap a card, give it the input its kind needs (text editor, image / audio /
video picker, number series), hit Run, see the result: text, images, detection
overlays, playable audio, stems, ranked hits, or a forecast chart.

First use of a card downloads its model; the progress bar is the process-wide
`CoreAI.onDownload` hook — one line in `OpsGalleryApp.init`.

The repo ships **zero binary assets**: sample text is a string literal, the sample
image is drawn with CoreGraphics at runtime, audio results are written to temp WAVs,
and everything else comes from your file picker.

## Run

```bash
cd Examples/OpsGallery
xcodegen generate
open OpsGallery.xcodeproj     # iPhone or Mac
```

`swift build` compiles the whole app headlessly on macOS (no Xcode) — the
compile-check door.

## Where the code is

- `Sources/OpRunner.swift` — the one switch that turns inputs into op calls; every
  case is exactly the line the [Cookbook](../../docs/COOKBOOK.md) documents.
- `Sources/GalleryView.swift` — the card grid over `Op.allCases`.
- `Sources/OpDetailView.swift` — input gathering + result rendering per op.
