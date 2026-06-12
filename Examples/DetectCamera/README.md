# DetectCamera

Real-time object detection from the live camera, fully on-device: `CameraFeed` streams
frames, `ObjectDetector` (RF-DETR, fp32, **no NMS**) turns each one into labeled boxes.

Measured on iPhone 17 Pro (Release, GPU): **nano 384² ≈ 27 ms / 36 FPS**,
**medium 576² ≈ 56 ms / 17 FPS**. First launch of a model pays a one-time on-device
specialization (~5 s nano), cached afterwards.

The whole ML surface is two calls:

```swift
let detector = try await ObjectDetector(model: .rfdetrNano)
for await frame in try await CameraFeed(framesPerSecond: 60).start() {
    detections = try await detector.detect(in: frame, scoreThreshold: 0.5)
}
```

## Run

```bash
xcodegen generate
open DetectCamera.xcodeproj
```

Run on an iPhone (camera required). First launch downloads the model from the Hugging
Face Hub ([mlboydaisuke/RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI));
later launches load from cache. The segmented control switches Nano ↔ Medium.

On launch the app also runs a small numerics gate against the bundled reference photo and
logs every confident detection (`GATE det …`) plus live timing windows (`STATS …`) — watch
them with `devicectl device process launch --console`.

### Sideloading models during development

`.aimodel` directories cannot ship inside the app bundle (the installer mistakes
extension-suffixed folders at the bundle root for nested bundles and rejects the app).
For development without the Hub, push them into the app's Documents instead:

```bash
xcrun devicectl device copy to --device <UDID> \
  --source rfdetr-nano_float32.aimodel \
  --destination "Documents/Models/rfdetr-nano_float32.aimodel" \
  --user mobile --domain-type appDataContainer \
  --domain-identifier com.coreaikit.detectcamera
```

The app prefers `Documents/Models/` over the Hub download. A bench hook
(`DETECT_VARIANT=nano|medium` in the launch environment) selects the initial model.
