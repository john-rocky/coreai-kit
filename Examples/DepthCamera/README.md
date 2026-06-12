# DepthCamera

Live monocular depth from the camera, fully on-device: `CameraFeed` streams frames,
`DepthEstimator` (Depth Anything 3, ~100 MB) turns each one into a relative depth map.

The whole ML surface is two calls:

```swift
let depth = try await DepthEstimator()
for await frame in try await CameraFeed(framesPerSecond: 5).start() {
    depthImage = try await depth.estimateDepth(for: frame).cgImage()
}
```

## Run

```bash
xcodegen generate
open DepthCamera.xcodeproj
```

Run on an iPhone (camera required). First launch downloads the model from the
Hugging Face Hub; later launches load from cache.
