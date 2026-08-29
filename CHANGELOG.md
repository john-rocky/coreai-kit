# Changelog

All notable changes to CoreAIKit are documented here. The project follows
[Semantic Versioning](https://semver.org) — pre-1.0, minor versions may change API,
patch versions never do. See [`docs/STABILITY.md`](docs/STABILITY.md) for the full
policy.

## [Unreleased]

### Changed

- **The model-fork pin moves to `0.2.3-zoo`** (was `0.2.2-zoo`) — in the manifest and in every
  lockfile: the 27 SwiftPM `Package.resolved` files and the SiriAsk Xcode workspace lockfile,
  which the 0.4.0 sweep did not reach and which still recorded `0.1.1-zoo`. `0.2.3-zoo` is
  upstream `main` through apple/coreai-models#207 (2026-08-28) with the zoo patches rebased on
  top; the tags before it sat on upstream at #90 (just after 0.2.0). The change a running app can see is
  upstream #121: the pipelined composite sampler reused one execution descriptor across
  in-flight steps and garbled output at temperature > 0 (word repetitions, doubled
  punctuation). No tag before `0.2.3-zoo` has that fix. One call site followed upstream #122's
  rename (`CoreAIRunner(from:)` → `CoreAIRunner(bundle:)`); nothing in the kit's own API moved.

### Fixed

- **Hybrid models kept their second turn.** `0.2.3-zoo` refuses a partial `reset(to:)` on
  models with recurrent state (Qwen3.5/3.6, LFM2.5, Granite 4, Nemotron-H — upstream #132
  throws `invalidState` where the earlier fork tags quietly fell back to a full reset), and
  the kit asks for exactly that rewind on every turn after the first. Reproduced on
  Qwen3.5-0.8B: turn 1 answered, turn 2 failed with *Partial reset is not supported for
  hybrid models with recurrent state*. `ChatSession` and the FoundationModels executor now
  rewind through one helper that falls back to a full reset when the engine says it cannot
  rewind, and report the tokens actually kept. A full reset costs a re-prefill of the
  prompt, nothing else — the kit feeds the full sequence either way.

## [0.4.0] — 2026-08-25

### Added

- **`CoreAI.tidyTranscript(_:)` + `KitTextNormalizer` — the other half of the dictation
  path.** The kit's ASR models all produce raw transcripts; nothing turned one into text a
  person would send. S1-mini by Superwhisper (catalog `s1-mini`, 796 MB, iPhone-verified)
  drops fillers, resolves false starts to whatever the speaker landed on, punctuates, and
  writes spoken numbers, dates, times, currency and email addresses out. Steered by the
  model's own three trained axes — `styling:` / `structure:` / `context:` — because it has no
  free-text instruction.

  It is deliberately **not** a flag on `proofread`, which is contracted to keep the wording as
  close to the original as possible: this op deletes, is English-only, and returns the empty
  string for filler-only input. And it does not ride `ChatSession`: that path sends an empty
  system prompt so every text op can share one model, while S1-mini's system prompt and
  control line are its trained input format, and `enable_thinking=False` is mandatory (leave
  thinking on and every call returns "" — a pipeline that looks like it works).

  **Long input is chunked and stitched**, which is the substance rather than a nicety: on iOS
  the shipped engine caps a growing KV cache at 1024 tokens, and the rewrite runs about as
  long as its input, so a whole meeting transcript passed in one call stops mid-sentence.
  Measured on iPhone 17 Pro — a 611-token transcript produced 413 tokens, every one
  token-identical to the Mac, then stopped at absolute position exactly 1024. Input is cut at
  word boundaries into ~450-token pieces, evenly rather than packed. `Examples/Tidy` is the
  runner (GUI + `tidy-cli`).

- **`CoreAI.capability(_:)` — the question the kit could not be asked.** `prepare` is an
  instruction ("fetch this now"); before an app can decide whether to *offer* a feature it
  needs "can this happen, and what will it cost the user". Answers `.ready`,
  `.needsDownload(bytes:)`, `.needsSystemAssets(what:)`, `.insufficientStorage(needs:free:)`
  or `.unsupportedDevice(reason:)` — offline, without touching the network, so a gallery of
  every op is not 23 round trips. `capabilities()` answers for all of them at once.

  `.needsSystemAssets` is new against the design: with transcription on Apple's backend, the
  honest answer for `transcribe` is neither "ready" nor a size — the OS may fetch a locale
  pack, and those bytes are shared with every app rather than charged to this one.

- **`coreai-doctor` — what will this app download.** `swift run coreai-doctor path/to/App`
  scans the sources for op calls and totals the models behind them, counting a model shared by
  three ops once. It asks `CoreAI.Op` rather than reimplementing the mapping, which is what
  keeps it right when a default changes — and the first thing it found was that `watch`,
  `watchDepth` and `scan` were missing from that enum, so an app using the live camera was
  being reported as having nothing to download.

- **`ModelStore.remoteSize(of:)` / `localSize(of:)` / `isCached(_:)` / `downloadPlan(for:)`** —
  sizes measured from the Hub rather than typed by hand, and the file list an adopter needs to
  hand to Apple's Background Assets. A Swift package cannot ship the downloader extension that
  framework requires; it can say exactly what to enqueue, which
  [`docs/BACKGROUND_ASSETS.md`](docs/BACKGROUND_ASSETS.md) walks through.

- **Errors for the three failures a shipped app actually hits** — `insufficientStorage`,
  `unsupportedDevice`, `systemAssetsUnavailable`. Every existing case was the vocabulary of
  someone who knows how a model repository is laid out (`notAHuggingFaceRepo`,
  `variantNotFound`), and none of them fire on the developer's machine, where the device is
  supported, the disk has room and the model is cached from the last run.

### Fixed

- **`Examples/OpsGallery` no longer built.** `watch`, `watchDepth` and `scan` joined
  `CoreAI.Op` (see `coreai-doctor` above) and the gallery's three switches over the enum were
  never extended, so the app stopped compiling. The grid now renders `CoreAI.Op.gallery` —
  everything except the streaming ops, which need a camera screen rather than a Run button and
  have their own examples — and the switches cover them explicitly, so the next streaming op
  cannot silently add a card whose button cannot work.

- **Every catalog VLM failed to load on iOS.** An iOS bundle built AOT holds one
  `*.aimodelc` directory and no `.aimodel`. `KitVisionModel` looked for `.aimodel` alone,
  found nothing, and fell back to a conventional `<name>.aimodel` — handing the runtime the
  path of a file that was never published (`Asset … is malformed: Missing hash file`). The
  only VLM that ran on a phone was Apple's own `SystemLanguageModel`.

  The same resolver had been written five times across the kit, and three copies were wrong:
  the ASR and audio encoders carried the identical fallback, and `KitDocReader` did not know
  `.aimodelc` at all. Those three had no iOS AOT bundle published yet, so they were waiting
  rather than working. All five now share `GraphBundle.resolve(in:)`, called inside
  `VLRuntime` / `ASRRuntime` / `AudioRuntime` so a new caller cannot miss it. AOT still wins
  over JIT when a bundle carries both — picking JIT pays on-device specialization every cold
  start, which is the cost the AOT build exists to remove.

- **A supported language in an unsupported region no longer hides transcription.** A device
  set to English-in-Japan reports `en_JP`, which is in nobody's supported-locale list, while
  ten English locales sit installed — strict matching declared the feature unsupported on a
  machine where it plainly worked. Resolution now falls back to the language, preferring the
  requested region, then the language's likely region via ICU (`en` → `en_US`), then anything
  installed. A language Apple genuinely cannot do is still refused rather than silently
  swapped for another. Found the first time this ran on a real machine.

### Changed

- **Transcription runs on Apple's on-device transcriber by default, at no download cost.**
  `CoreAI.transcribe` used to pull Whisper — 3.2 GB on an iPhone — for a capability iOS 27
  ships in `SpeechAnalyzer` + `SpeechTranscriber`, with OS-managed locale assets shared
  across apps. This package's own porting contract has a gate for exactly that ("Apple's
  stock stack does not already ship this capability. If it does, stop") and speech-to-text
  was the clearest case of it in the catalog. Measured on this machine: 45 supported locales,
  ten already installed, a 3.9 s clip transcribed in 0.18 s.

  `options: .model("whisper-large-v3-turbo")` opts back in, for a locale Apple lacks on the
  device in hand, behaviour that must not change under an OS update, or an offline guarantee.

- **`CoreAI.transcribeMeeting` is now 238 MB rather than 3.4 GB.** Who spoke when is the one
  speech capability Apple ships nothing for — `Speech.framework` has no speaker API at all —
  so Sortformer is a real download and the transcription it is paired with is free. The new
  `SpeechToText` protocol is the seam: `SystemTranscriber` and `KitTranscriber` are
  interchangeable, and `MeetingTranscriber(asr:)` takes either.

  Text-to-speech is deliberately **not** changed. Apple's voices are free but cannot be
  cloned, so the catalog TTS models still do something the system cannot, and defaulting them
  away would leave a wrapper with no reason to exist.

### Added

- **`VoiceActivityDetector`** — where speech starts and stops, which the speech design calls
  the only genuinely new component behind `listen()`. Everything else in the live speech path
  is composition of models that exist; deciding that a pause ended an utterance is not
  something a model does, and Apple's nearest offering (`SNClassifySoundRequest`) is a
  classifier over second-long windows, not an endpointer. Energy against an adaptive noise
  floor, with hysteresis, a hangover across stop consonants, and `minimumSilence` exposed
  because it is the latency choice and there is no correct value for it. No model, no
  dependency. `segments(in:)` is useful with no microphone in sight: it cuts a two-hour
  recording at pauses, which is how you feed a model whose window is thirty seconds.

  Two things the tests changed. `minimumSpeech` was being applied to the *padded* clip, so a
  40 ms click became 440 ms of "speech" and passed the filter that exists to reject it — the
  padding is for the model's benefit and must not decide what counts. And an utterance could
  stay open indefinitely: an energy detector cannot tell a fan from a voice, so a step in room
  level held one open forever and what eventually reached the model was longer than its
  window. `maximumSpeech` bounds it, splitting mid-word rather than handing over a clip
  nothing can take.

- **`KitTracker`** — detections in, stable identities out. A detector is stateless by
  construction, so "is that the same person as last frame" is a question no model answers and
  Apple ships nothing for; every counting, dwell-time or follow-the-object feature needs it
  and rebuilds it. Hungarian association (optimal, not greedy — greedy swaps identities
  exactly when two objects cross), constant-velocity prediction against real elapsed time,
  and a second association pass over low-confidence detections so a partly-occluded object
  keeps its id. No model, no framework, 18 tests.

  Two bugs the tests caught, both of which would have shipped: assignment crashed whenever
  there were more tracks than detections — which is every frame where something leaves — and
  IoU-only association fell apart at low frame rates. The second one matters most: box
  overlap between consecutive frames drops from 0.82 at 30 fps to 0.25 at 5 fps for the same
  motion, and 5 fps is what this package's own thermal governor produces on a hot phone. It
  worked on the desk and would have assigned a new id to every object in the user's hand.
  Association now falls back to centre distance, gated by how far the object could actually
  have travelled in the elapsed time.

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

### Changed

- **Parakeet is no longer described as something Apple does not have.** `apple/coreai-models`
  merged its own Parakeet TDT 0.6B v3 export on 2026-08-07 (#136) and live streaming over it
  on 08-21 (#184), in a package that builds for iOS as well as macOS. Three places here —
  `docs/TASK_MAP.md`, `docs/SPEECH_API.md` and `Examples/Transcribe/README.md` — argued from
  a macOS-only Parakeet, and one of them called an iOS port "the cheapest high-value work on
  the map". Corrected in place with the date, rather than deleted: the catalog entry is
  still a macOS-only bundle and that is still worth saying out loud. Kokoro is the half of
  that argument that survives.

### Fixed

- **0.3.0 does not compile against the current Xcode 27 SDK; this release does.**
  FoundationModels renamed `LanguageModelCapabilities.init(capabilities:)` to `init(_:)`
  between Xcode 27 beta 3 and beta 5, which is five call sites here and one in the model
  fork. `docs/STABILITY.md` tells adopters to depend on a tag because shipping apps depend
  on this package — so between 2026-08-16 and today, the documented way to adopt CoreAIKit
  was the one way that failed, and it failed at compile time with an error naming Apple's
  type rather than anything of ours. `main` has been buildable since; nothing was tagged.
  If you are pinned to `from: "0.3.0"`, move to `0.4.0`.

- **Three examples were resolving Apple's package, not the patched fork.** `DocChat`,
  `FMToolDemo` and `GuidedDemo` carried a `Package.resolved` pinning
  `apple/coreai-models` 0.1.0 — a lockfile from before the fork existed. Every example
  depends on the kit by path, so the kit's own `exact:` requirement won them back on the
  first resolve; nobody had run one. All 25 lockfiles now record `0.2.2-zoo`.

- **CI can no longer pass by pinning an Xcode nobody uses.** Every workflow checked that
  `$DEVELOPER_DIR` was a directory and nothing else. At GA that check stops meaning
  anything: the release Xcode installs alongside the beta, the beta's folder keeps
  existing, and the gates keep publishing green for an SDK no shipping app is built with.
  `.xcode-pin` now names path *and* ProductBuildVersion in one place, and
  `scripts/check-xcode-pin.sh` refuses to run the job unless the installed build is the
  one the pin claims and no release build of the same Xcode train has appeared beside it.

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
