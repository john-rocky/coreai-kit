# Changelog

All notable changes to CoreAIKit are documented here. The project follows
[Semantic Versioning](https://semver.org) — pre-1.0, minor versions may change API,
patch versions never do. See [`docs/STABILITY.md`](docs/STABILITY.md) for the full
policy.

## [Unreleased]

### Added

- **`CoreAI.watch()` / `watchDepth()` — the camera as an op.** `CameraFeed` has vended frames
  since the beginning and no op consumed them, so every live example wrote its own two-stage
  task pipeline, stale-frame policy and stats window. That loop is now `LivePipeline`
  (CoreAIKitCore, Foundation only) with `LiveVision` over it, and `CoreAI.watch()` on top:
  detections per frame, normalized boxes, `result.stats` carrying *measured* frame rate,
  median latency, dropped frames and thermal state. Attach an `AVCaptureVideoPreviewLayer`
  to `watch.captureSession` and the preview costs nothing.

- **Thermal governor, on by default.** A sustained camera-plus-model loop is the hottest
  thing an app can do on a phone. `LiveGovernor` halves the target frame rate at a
  `.serious` thermal state and quarters it at `.critical` — `thermalBackoff: 1` opts out,
  which is appropriate only for a bench run.

- **Event triggers — `CoreAI.watch(for:)`.** A small detector runs continuously and decides;
  the expensive model runs on the frames that matter. `.label("person")`, `.anything()`, or
  `.when { … }`, each carrying a cooldown, because a predicate over a live stream is true for
  as long as the object is in shot. Yields a rendered `CGImage` only when it fires.

- **`KitDetector` real-time path** — `prepare(_:)` / `detect(_:)` over a 32BGRA capture
  buffer, plus `inputSize`. `Examples/DetectCamera` hand-wrote an enum to put RF-DETR and
  YOLOX behind one prepare/detect surface; that abstraction now lives in the package where
  it belongs.

- **`CoreAI.scan(videoAt:)` — a video file as a timeline.** `recognizeAction` answers one
  question about a whole clip; an app holding an hour of footage usually wants to know
  *where*. `scan` samples, runs a model per sample, and stamps the results; `scan(videoAt:for:)`
  is the offline twin of `watch(for:)`, with the cooldown counted in video time. Unlike the
  live path a scan drops nothing — it was asked for a specific set of samples and delivers
  all of them.

- **`VideoFile.stream` picks its reader from the sample rate.** Seeking costs per sample,
  sequential decode costs per clip, and the two cross over near one sample per second of
  30 fps source. Measured on an M4 Max over a 60 s clip: at 15 samples/s sequential is
  **17× faster** (0.48 s vs 8.20 s), at 0.1 samples/s seeking is **4.2× faster** (0.11 s vs
  0.47 s). `.automatic` chooses; `.seeking` / `.sequential` override. The sequential path
  rides OS 27's `AVAssetReader.outputProvider`, so it suspends rather than blocking a thread.

- **Scene-change sampling.** `minimumChange` skips frames too similar to the last one kept
  (mean absolute difference on a 32×32 grey thumbnail). On an 8 s clip that is a slow zoom
  over one photo, 2 samples/s runs the detector 16 times and `--changes` runs it once.

- **`Examples/LiveCamera`** — the four live tasks as four tabs on an iPhone, with the
  measured stats and the thermal governor visible on screen, plus `swift run live-cli` for
  the offline half (video scan and the preprocessing benchmark) with no device.

### Fixed

- **Live depth no longer renders a bitmap per frame.** `LiveVision.depth` fed the estimator
  through `CIContext.createCGImage`; it now takes the capture buffer directly through the new
  `PixelBufferPreprocessor` (vImage scale + vDSP channel split, scratch reused). Measured on
  an M4 Max, 640×480 → 224², release build: **0.13 ms against 0.78 ms**, about 6×. The two
  paths agree bit-for-bit on flat colour, so this is not a change in what the model sees —
  only in what it costs to hand it over. `DepthEstimator` gains the matching
  `prepare(_:)` / `estimateDepth(_:)` real-time pair, and `ObjectDetector`'s previously
  private fast path is now the shared implementation.

- **Op models are now evictable.** `OpModels` and its six siblings cached every load and
  never released one, so an app calling three ops in sequence kept three models resident and
  was jetsammed on a phone — the capacity planning the op layer exists to remove. All
  thirteen caches now share `ResidentCache`, admitting loads through a process-wide
  `ModelResidency`: least-recently-used models are dropped to make room, a model an op is
  currently running on is never dropped, memory pressure drops everything idle, and a
  re-load after eviction is a cache miss rather than an error. `CoreAI.evictModels()` and
  `CoreAI.residentModels()` expose it.

## [0.3.0] — 2026-07-28

### Added

- **iOS dynamic-KV guard** — the engine (coreai-models `0.2.1-zoo`) now caps a pipelined
  turn's KV pre-grow at capacity 1024 on iOS for dynamically-sized-KV bundles: the on-device
  compiler miscompiles those specializations at seq ≥2048 (corrupt output from token 1).
  ChatDemo additionally clamps `maxResponseTokens` to 896 on iOS as belt-and-suspenders.
  Tracked in [#5](https://github.com/john-rocky/coreai-kit/issues/5); upstream
  apple/coreai-models#124. Both guards come out when the compiler fix lands.

- **Nanbeige4.2 3B** (`nanbeige4.2-3b`) — chat, the catalog's **first community-contributed
  model**: ported and published by [@ukint-vs](https://github.com/ukint-vs)
  ([zoo PR #6](https://github.com/john-rocky/coreai-model-zoo/pull/6)), served from the
  contributor's HF repo at a pinned revision. A recurrent Llama — 22 physical blocks execute
  twice per token (44 KV-cache layers) — int8, 4.6 GB, `thinking` chat template. Device-gated
  on iPhone 17 Pro: token-exact vs the fp32 oracle (24/24 ×2 runs), 8.5 prefill / 6.4 decode
  tok/s settled (two-pass models pay ~2× weight traffic per token; Mac M4 Max: 57 tok/s).

- **`KitSeparator`** — music source separation (Mel-Band RoFormer, Kim Vocal): a song in, a
  vocals stem and an instrumental stem out. `separate(_:)` on a decoded stereo mix or
  `separate(contentsOf:)` on any file AVFoundation reads. New catalog kind `separation`
  (`melband-roformer-vocal`). iPhone 17 Pro: an 8 s chunk in 1.23 s (~6.5x real-time).
- **`KitDialogue`** — multi-speaker / dialogue text-to-speech (VibeVoice-Realtime-0.5B), the first
  kit API that performs a *script*: `perform("Speaker 1: …\nSpeaker 2: …")` renders each turn from
  its own voice preset and concatenates them. 25 packaged voices, free text (Qwen2.5 tokenizer +
  mmapped fp16 embedding table — no torch, no 272 MB read). `KitSpeaker(catalog:)` also accepts
  the id and speaks one line in the default voice. Pairs with `KitDiarizer` for a
  generate → diarize loop.
- `AudioFile.pcmStereo(_:sampleRate:)` — decode any audio file to stereo at a chosen rate (the
  separation models need the full band, not the 16 kHz speech downmix).
- `StatefulGraphModel.seedState(keys:values:prefillLength:)` — seed a decode bundle's KV cache from
  an **external** prefill (a voice prompt, a cached prefix) instead of replaying it.

- **[`SECURITY.md`](SECURITY.md)** — the trust boundary written down: two hosts (GitHub raw for
  the catalog, Hugging Face for the weights), revision pins as the integrity mechanism, and the
  parts that are absent — `catalog.json` is unsigned, downloads carry no independent checksum
  manifest, bundles are not code-signed. Includes the escape hatch for production apps:
  `ModelCatalog.load(from:)` takes a URL, so you can host the catalog yourself.

- **[`AGENTS.md`](AGENTS.md)** — the contract for coding agents building on the kit: which layer
  to reach for (Apple's Foundation Models first), the catalog as data to read rather than guess
  at, the failures that only appear on a real device, and an accurate statement of what
  "verified" covers.

### Fixed

- **The offline fallback catalog resolved unpinned revisions.** `ModelCatalog.load()` falls back
  to the compiled-in `ModelCatalog.builtin` when the network fetch of `catalog.json` fails, and
  that literal carried no revision pins — while `CatalogEntry.modelID(path:)` resolves
  `revision ?? "main"`. A network failure therefore downgraded a shipped app from the exact bytes
  that were gated to whatever the model repository's `main` branch held at that moment, which is
  the opposite of what the catalog documents. Pins are now generated from `catalog.json` into
  `BuiltinPins.swift` and overlaid onto the literal; CI fails if the two drift, and
  `BuiltinPinsTests` fails if any entry reaches `builtin` without a pin. **Apps pinned to 0.2.0
  or earlier are affected** — the fallback path is the one that matters here, so an app that has
  never seen a catalog fetch failure has never hit it.

## [0.2.0] — 2026-07-10

### Added

- **Catalog revision pinning.** Every `catalog.json` entry now carries a `revision`
  (a Hugging Face commit hash), and every `(catalog:)` initializer downloads from
  exactly that revision — including multi-bundle models (VL towers, TTS glue, OCR
  assets, Gemma PLE tables, ASR encoder pairs). A push to a model repo can no longer
  change what apps receive; pins are bumped deliberately by re-running
  `scripts/pin-catalog.py` and committing the catalog change.
- `CatalogEntry.revision`, `CatalogEntry.modelID(path:)`, and `pinned(_:)` on
  `ModelID`, `GemmaModelID`, and `VLModelID`. Older catalogs without pins keep
  resolving `main`; older package versions ignore the new field — the change is
  backward and forward compatible.
- `scripts/pin-catalog.py` — re-pins all entries to each repo's current head
  (`--check` verifies every entry is pinned and reports drift, for CI).
- CI: build + unit tests on every push/PR, plus a nightly gate that fetches the live
  catalog, downloads a pinned model, and generates on the GPU end-to-end
  (self-hosted runner, macOS 27 beta + Xcode 27 beta).

### Fixed

- macOS build with current Xcode 27 betas: `os_proc_available_memory()` is now
  compiled only on iOS — the newer SDK marks it unavailable on macOS, which broke
  `swift build` for Mac consumers. Semantics are unchanged (macOS always owned
  tables).
- `catalog.json` / built-in snapshot drift: `nemotron-3.5-asr-streaming-0.6b` was
  missing from the built-in snapshot, while `mineru2.5-pro`, `glm-ocr`, and TimesFM's
  `static-shape` engine hint were missing from the shipped `catalog.json`. The sync
  test now pinpoints per-entry drift and requires every shipped entry to carry a pin.

### Changed

- `KitASRModel`, `KitWhisperModel`, `KitParakeetModel`, and `KitForecaster`
  `(catalog:)` initializers now consult the live catalog like every other surface, so
  unknown ids and wrong-platform requests fail with the same clear, early errors.
- First run on 0.2.0 re-downloads models once: pinned bundles cache under their
  revision (`<repo>/<revision>/<variant>`) instead of `main`.

## [0.1.0] — 2026-07-07

Initial tagged release: `ModelStore` + `ModelCatalog`, `ChatSession` (streaming,
stats, guided generation), FoundationModels providers (text / VL / Gemma), vision
pipelines (CLIP, depth, detection, super-resolution, video), embeddings + retrieval,
ASR / TTS / diarization / OCR / audio-QA / diffusion-LM / forecasting surfaces, and
SwiftUI components.
