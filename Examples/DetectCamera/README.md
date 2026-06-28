# DetectCamera

Real-time object detection from the live camera, fully on-device, over the zero-copy
capture path — two detector families behind one picker: **RF-DETR** (DETR, fp32, **no
NMS**, `ObjectDetector`) and **YOLOX-S** (Megvii's dense anchor-free detector, fp32,
**host NMS** + letterbox/BGR preprocessing, `YOLOXDetector`). Architecture follows the
fastest known iOS detection-app pattern:

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

Run on an iPhone (camera required). First launch downloads the RF-DETR model from the
Hugging Face Hub ([mlboydaisuke/RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI));
later launches load from cache. The segmented control switches between the variants
(Nano / Medium / Seg / **YOLOX**). YOLOX-S downloads from the Hub
([mlboydaisuke/YOLOX-CoreAI](https://huggingface.co/mlboydaisuke/YOLOX-CoreAI)) like the
others; it gates fp32-clean and runs **4.8 ms / 208 FPS on the M4 Max GPU** and
**~22 ms / 35–40 FPS end-to-end on the iPhone 17 Pro** (device-verified live, GPU,
Low Power Mode on; the static graph specializes on-device in ~2.6 s on first load).

On launch the app also runs a small numerics gate against the bundled reference photo and
logs every confident detection (`GATE det …`) plus live timing windows (`STATS …` with
median inference ms and end-to-end wall FPS) — watch them with
`devicectl device process launch --console`.

### Mask rendering (vectorized)

Masks follow the yolo-ios-app rendering recipe: `InstanceMask` stores raw
logits (foreground ⇔ logit > 0 ≡ sigmoid > 0.5, so binarization needs no
transcendental math at all), and `MaskRenderer` composites every instance into
one premultiplied-RGBA bitmap with vDSP — binarize via `vDSP_vlim`, blend
`plane + m·(color − plane)` per channel, one strided float→u8 pass at the end.
No per-pixel Swift loops; scratch buffers are reused across frames, so the
cost stays sub-millisecond from seg-nano's 78² grid up to seg-2xlarge's 192².
Device-verified: identical gate mask pixel counts before/after vectorization.

### Capture-rate tuning (measured)

Capture rate trades against inference latency — the ISP and preview compete with the
GPU. On iPhone 17 Pro with nano, **true 60 fps capture + the two-task pipeline is the
sweet spot (33–39 FPS)**; 30 fps capture phase-locks the loop (a ~27 ms model rides the
33 ms frame grid and can drop to every other frame). Three traps worth knowing: a
`sessionPreset` commit resets custom frame durations (configure the device *after*
`commitConfiguration`); the preset's default 720p format tops out at 30 fps, so
`CameraFeed` switches `activeFormat` to the 60 fps sibling; and the
kCVPixelBufferWidth/Height data-output scaling request (`dataOutputSize`) is silently
ignored on iOS 27 beta — buffers arrive at full session resolution (verified by
dumping a frame), so don't attribute wins to it. Throughput numbers move with chassis
temperature; compare runs back-to-back with cooldowns. Bench/debug hooks
(launch environment): `DETECT_VARIANT=nano|medium`, `DETECT_FPS=<n>`,
`DETECT_TESTBOX=1` (replace detections with corner/center reference boxes to
validate overlay mapping via screenshot), `DETECT_DUMP=1` (write the first
delivered buffer to Documents/framedump.bin for host-side inspection).

### ANE mode (experimental, measured honestly)

The UNIT picker can request the Neural Engine. Today, on iOS 27 beta, it buys
nothing: a monolithic graph under `.neuralEngine` preference falls back to the
GPU delegate entirely (the gather-heavy deformable head is not ANE-lowerable),
and even a **split deployment** — separate `rfdetr-<v>_backbone.aimodel`
(pure ViT) on `.neuralEngine` chained into `rfdetr-<v>_head.aimodel` on
`.gpu` — still executes the backbone on the GPU (fingerprint: detections
identical to the GPU run to 3 decimals, no ANE-compile pause on first load,
and timing equal to GPU at the same thermal state, +~5 ms two-graph handoff).
Same-thermal comparison (nano, thermal=fair): GPU monolith 24.7–25.6 ms /
33–40 FPS vs split-"ANE" 29.5–30.6 ms / 24–30 FPS.

The split infrastructure (`ObjectDetector(backboneAt:headAt:)`) stays — it is
the right shape for when the runtime starts honoring ANE placement for these
graphs (or for an AOT `coreai-build compile --unit neural-engine` backbone).
ANE would matter for thermals: the GPU at `thermalState=serious` throttles
nano 25 → 75–103 ms.

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

**YOLOX-S** is on the Hub too, so the app downloads it automatically; to sideload a local
build instead (the gated `yolox-s_float32.aimodel` is staged in `Models/`):

```bash
xcrun devicectl device copy to --device <UDID> \
  --source Models/yolox-s_float32.aimodel \
  --destination "Documents/Models/yolox-s_float32.aimodel" \
  --user mobile --domain-type appDataContainer \
  --domain-identifier com.coreaikit.detectcamera
```
