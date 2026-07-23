# Changelog

All notable changes to CoreAIKit are documented here. The project follows
[Semantic Versioning](https://semver.org) — pre-1.0, minor versions may change API,
patch versions never do. See [`docs/STABILITY.md`](docs/STABILITY.md) for the full
policy.

## [Unreleased]

### Added

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
