# ActionCamera

Live action recognition from the camera, fully on-device: `CameraFeed` streams frames, a
rolling 16-frame clip goes through `ActionRecognizer` (V-JEPA 2 ViT-L, SSv2 head — Meta's
video world model) and the top actions come back with confidences.

The whole ML surface is two calls:

```swift
let recognizer = try await ActionRecognizer(catalog: "vjepa2-vitl-ssv2")
let actions = try await recognizer.classify(videoAt: clipURL)   // or classify(frames:)
```

## Run

```bash
xcodegen generate
open ActionCamera.xcodeproj
```

Run on an iPhone (camera required). First launch downloads the model from the
Hugging Face Hub; later launches load from cache.

## CLI (macOS, no Xcode)

```bash
swift run action-cli --video sample.mp4
```

`sample.mp4` is a synthetic clip (a hand pushing a block from left to right) so the
command works out of the box; point `--video` at a real clip for real footage.
