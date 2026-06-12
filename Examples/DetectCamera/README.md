# DetectCamera

Real-time object detection from the live camera, fully on-device: RF-DETR (fp32, **no
NMS**) over the zero-copy capture path. Architecture follows the fastest known iOS
detection-app pattern:

- **`AVCaptureVideoPreviewLayer` renders the camera directly** — the compositor shows
  the full-rate feed; the app never converts a frame for display.
- **`CameraFeed.startPixelBuffers()`** hands the model raw 32BGRA buffers
  (hardware-scaled to ~model size via `dataOutputSize`), and
  **`ObjectDetector.prepare`/`detect`** split CPU preprocessing (vImage scale + vDSP
  channel split — no CGImage/CIContext anywhere) from GPU inference so frame N+1
  preprocesses while frame N runs.
- `bufferingNewest(1)` streams drop stale frames instead of queueing; UI updates are
  fire-and-forget hops off the inference path.

Measured on iPhone 17 Pro (Release, GPU, 60 fps capture): **nano 384² ≈ 25 ms /
33–39 FPS end-to-end**, medium 576² ≈ 63 ms / ~15 FPS. Sustained max-load throughput
drops on a hot chassis (thermal); the first launch of a model pays a one-time
on-device specialization (~5 s), cached afterwards.

The whole ML surface:

```swift
let detector = try await ObjectDetector(model: .rfdetrNano)
for await frame in try await feed.startPixelBuffers() {
    detections = try await detector.detect(in: frame.pixelBuffer, scoreThreshold: 0.5)
}
```

(the example splits `prepare`/`detect` across two tasks for the extra few ms.)

## Run

```bash
xcodegen generate
open DetectCamera.xcodeproj
```

Run on an iPhone (camera required). First launch downloads the model from the Hugging
Face Hub ([mlboydaisuke/RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI));
later launches load from cache. The segmented control switches Nano ↔ Medium.

On launch the app also runs a small numerics gate against the bundled reference photo and
logs every confident detection (`GATE det …`) plus live timing windows (`STATS …` with
median inference ms and end-to-end wall FPS) — watch them with
`devicectl device process launch --console`.

### Capture-rate tuning (measured)

Capture rate trades against inference latency — the ISP and preview compete with the
GPU. On iPhone 17 Pro with nano: 30 fps capture phase-locks the loop to ~30 FPS;
**60 fps capture + hardware-scaled data buffers is the sweet spot (33–39 FPS)**; 60 fps
*without* `dataOutputSize` slows inference 25→39 ms and LOWERS throughput. Two traps
worth knowing: a `sessionPreset` commit resets custom frame durations (configure the
device *after* `commitConfiguration`), and the preset's default 720p format may top out
at 30 fps (switch `activeFormat` to the 60 fps sibling). Bench/debug hooks
(launch environment): `DETECT_VARIANT=nano|medium`, `DETECT_FPS=<n>`,
`DETECT_TESTBOX=1` (replace detections with corner/center reference boxes to
validate overlay mapping via screenshot), `DETECT_DUMP=1` (write the first
delivered buffer to Documents/framedump.bin for host-side inspection).

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

The app prefers `Documents/Models/` over the Hub download.
