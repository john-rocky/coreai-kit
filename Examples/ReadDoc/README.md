# ReadDoc

On-device document OCR: pick an image of a document, `KitDocReader` (Unlimited-OCR,
~4.5 GB) turns it into structured markdown — tables survive as `<table>/<tr>/<td>` markup.

The whole ML surface is two calls:

```swift
let reader = try await KitDocReader(catalog: "unlimited-ocr")
let markdown = try await reader.read(imageAt: documentURL)
```

## Run

```bash
xcodegen generate
open ReadDoc.xcodeproj
```

Run on a Mac (the shipped bundle is macOS-only today). First launch downloads the model
from the Hugging Face Hub; later launches load from cache.

## CLI (macOS, no Xcode)

```bash
swift run readdoc-cli --image sample.png
```

`sample.png` is a small rendered document (heading, paragraph, table) so the command works
out of the box.
