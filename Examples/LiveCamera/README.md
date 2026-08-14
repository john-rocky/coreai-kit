# LiveCamera

Four live tasks, one per tab, each a few lines over `CoreAIOps`. The point of the example is
what is *not* in it: no capture session wiring, no frame-drop policy, no stats window, no
thermal check. Those live in the kit — see [`Sources/QuickStart.swift`](Sources/QuickStart.swift),
which is the whole take-home core and imports no UI.

| Tab | Call | Model | First-use download |
|---|---|---|---|
| **Detect** | `CoreAI.watch()` | RF-DETR nano / YOLOX-S (picker) | 103 MB / **36 MB** |
| **Depth** | `CoreAI.watchDepth()` | Depth Anything 3 Small | **54 MB** |
| **Trigger** | `CoreAI.watch(for: .label("person"))` | detector, plus Qwen3-VL on demand | 103 MB (+ 3.3 GB if you tap Describe) |
| **Scan** | `CoreAI.scan(videoAt:)` | RF-DETR nano | 103 MB |

Detect, Depth and Scan are ordinary app assets. The VLM behind *Describe* is not, which is
why it is a button rather than something the trigger does by itself.

## Run it on a device

The CoreAI framework is not in the iOS Simulator SDK, so this needs a real iPhone.

```
xcodegen generate     # only if you change project.yml; the .xcodeproj is committed
open LiveCamera.xcodeproj
```

Pick your team in Signing & Capabilities (or `export DEVELOPMENT_TEAM=XXXXXXXXXX` before
generating), select your iPhone, Run.

## Run the offline half with no device

The camera tabs need a phone. The video scan is the same pipeline fed from a file, so it runs
headless on macOS:

```
swift run live-cli --video clip.mov --fps 2          # a detection timeline
swift run live-cli --video clip.mov --fps 2 --changes # skip frames that barely moved
swift run live-cli --video clip.mov --for person      # only the moments that matched
swift run -c release live-cli --bench                # the preprocessing measurement
```

On an 8 s clip that is a slow zoom over one photo, `--fps 2` runs the detector 16 times and
`--changes` runs it **once** — the frames after the first are not different enough to be worth
a model call. That is the entire difference on fixed-camera footage.

## What to look at while it runs

**The stats badge.** It reports what the pipeline achieved, not what it was asked for:
measured frame rate, median model latency, frames dropped since the last result, and the
thermal state. On a phone the requested rate is not the interesting number.

**What happens when the phone gets hot.** Leave Detect running. As the thermal state reaches
`serious` the badge says `hot · governed` and the frame rate halves — on purpose. A sustained
camera-plus-model loop is the hottest thing an app can do, and a feature that flattens the
battery is one the shipping team removes. `LiveVision.Options(thermalBackoff: 1)` turns it
off, which is right for a benchmark and wrong for a product.

**The model picker on Detect.** Switching RF-DETR to YOLOX-S changes one catalog id. The two
have different graph contracts — one needs host-side NMS and the other does not, one takes
letterboxed BGR and the other square RGB — and none of that reaches this app.

## Live drops, offline does not

The camera tabs throw frames away when the model falls behind, because a stale overlay is
worse than a missing one. The Scan tab delivers every sample it was asked for, however long
that takes, because nothing is on screen waiting. Same two stages, opposite policy — worth
knowing which one you are in.
