# Versioning & stability

CoreAIKit is meant to be **depended on** — by apps that ship. This page says exactly
what that promise covers today.

## Semantic versioning

Releases follow [SemVer](https://semver.org). Pre-1.0:

- **Patch** (`0.x.Y`): bug fixes and additive changes only. Always safe to update.
- **Minor** (`0.X.0`): may change or remove API. Anything breaking is called out in
  [`CHANGELOG.md`](../CHANGELOG.md) with a migration note.

Pin the package with `from: "0.2.0"` (or `exact:` if you want zero surprises) — every
release is a tag on this repo.

## The catalog contract

The live catalog (`catalog.json`, fetched at runtime with the built-in snapshot as
offline fallback) is a public contract:

- **Ids are forever.** A catalog id (`"qwen3.5-2b"`) is never repurposed for a
  different model. Entries may gain variants; removals are announced in the
  changelog first.
- **Entries are pinned.** Each entry carries a `revision` — the Hugging Face commit
  hash of the verified artifact. What you download is what was tested on real
  hardware, byte for byte, until the pin is deliberately bumped. Pins are refreshed
  with `scripts/pin-catalog.py`; CI fails if an entry ships unpinned.
- **New kinds are forward-compatible.** A catalog entry whose `kind` this package
  version doesn't know decodes cleanly and is filtered out — a newer catalog never
  breaks an older app.

## Surface stability

| Surface | Status |
|---|---|
| `ModelStore`, `ModelCatalog`, `ModelID`, `ChatSession`, `KitLanguageModel` | Stable — breaking changes only in a minor release, with migration notes |
| `CoreAIKitVision` (`GraphModel`, detectors, depth, CLIP), `CoreAIKitEmbeddings` | Stable |
| Newer capability surfaces (VLM, TTS, ASR, diarization, OCR, dLLM, forecasting, audio-QA) | Settling — APIs may still be reshaped as more models of each kind land; changes are changelogged |
| Anything `internal` or underscore-prefixed, `Examples/` | No guarantees |

## Verification

Every model in the catalog is verified on real hardware (Apple silicon Mac and, where
published, iPhone) before its pin lands — parity against the reference
implementation and measured tokens/sec, recorded in the
[model zoo](https://github.com/john-rocky/coreai-model-zoo). CI builds and tests
every push on macOS 27 beta; a nightly gate downloads a pinned model from the live
catalog and generates on the GPU end-to-end.
