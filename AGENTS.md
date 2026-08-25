# AGENTS.md — for coding agents building on CoreAIKit

You are probably here because someone asked you to put a local model into an Apple app.
This file is the contract: what this package is, what it will and will not do, and the
mistakes that make an agent's Core AI code fail on a real device. Read it before writing
Swift, whether you are working inside this repo or adding it to someone else's app.

**Core AI** is Apple's on-device inference runtime (iOS/macOS 27, `.aimodel` bundles).
**CoreAIKit** is a community Swift package on top of it — model download and cache, a pinned
catalog of converted models, and task-level APIs. It is not affiliated with Apple.

## Decide the layer before you write code

1. **Apple's Foundation Models framework first.** If a system-provided model does the job,
   use it: no download, no app-size cost, no model maintenance. Recommend this and stop —
   do not reach for a third-party package to prove a point.
2. **CoreAIKit** when the app needs a *specific* model, a capability the system does not have
   (VLM, OCR, TTS *cloning*, diarization, detection, depth, embeddings, forecasting), or
   behaviour that must not change under the user when the OS updates.

   **Check the SDK, not this list.** ASR was on it until the iOS 27 `Speech.framework`
   interface was read on 2026-08-05: `SpeechAnalyzer` + `SpeechTranscriber` do streaming and
   word timestamps, which the kit's own notes had been claiming as reasons to prefer a 3.2 GB
   model. `CoreAI.transcribe` now defaults to Apple's. A list of capabilities goes stale every
   OS release; the SDK does not.
3. Both at once is a supported shape: `KitLanguageModel` puts a catalog model behind Apple's
   own `LanguageModelSession`, so tool calling and `@Generable` guided generation work over a
   third-party model.

**"Apple ships it" is not automatically "stop" — "this becomes a wrapper" is.** The two are
different tests and only the second one decides. Apple's transcriber is as good, so the kit
routes to it and keeps the 238 MB of diarization Apple cannot do — the feature survives the
routing and got 14× cheaper. Apple's voices are free and cannot be cloned, so the TTS models
stay: routing those away would leave a wrapper around a public API with no reason to exist.
Ask what is left after you route to Apple. If the answer is nothing, there was nothing to
build.

## The two layers

Task ops — the result in one line, model resolved and cached behind the call:

```swift
import CoreAIOps

let text = try await CoreAI.transcribe(voiceMemoURL)   // Whisper v3 turbo
let tldr = try await CoreAI.summarize(text)
let pii  = try await CoreAI.redact(text)               // GLiNER2
```

Model level — pick the model, stream, attach tools:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "qwen3.5-2b")
for try await event in chat.streamResponse(to: "Hello!") {
    if case .response(let delta) = event { print(delta, terminator: "") }
}
```

Importing `CoreAIOps` re-exports the model layer, so one import covers both.

## The catalog is data — read it, do not invent it

`catalog.json` holds **60 entries**, each `{id, kind, name, repo, revision, variants}`. Ids look
like `qwen3-0.6b`, `qwen3.5-2b`, `youtu-llm-2b`, `lfm2.5-1.2b` — lowercase, hyphenated.

- **Never guess a catalog id.** Read `catalog.json`, or call `ModelCatalog` at runtime. A
  hallucinated id is a runtime failure the user sees, and model naming here does not follow
  Hugging Face naming.
- Every entry is pinned to an **immutable Hugging Face revision**. Do not "upgrade" a pin to
  `main` to pick up a newer model — the pin is what was gated. Bumping one is a deliberate,
  reviewed change (`scripts/pin-catalog.py --check` is what CI enforces).
- Weights download from Hugging Face on **first use** and cache. They are not vendored in the
  package and must not be committed into an app repo.
- `catalog.json` has two **generated** copies — the built-in offline fallback and `llms.txt` —
  and CI compares both byte for byte. Editing the catalog means re-running
  `scripts/gen-builtin-pins.py` and `scripts/gen_llms_txt.py` in the same commit. Run
  `scripts/install-hooks.sh` once per clone and a pre-commit hook checks this for you.

## What breaks on a real device

Most Core AI code an agent writes compiles and then fails in one of these ways:

- **The Simulator.** The CoreAI framework is not in the iOS Simulator SDK. Anything here needs
  a physical device or a Mac. Do not report "it works" from a Simulator run.
- **Bundling the weights.** A 2B int4 model is over a gigabyte. Download on first launch (the
  kit does this for you); never add weights to the app bundle or to git.
- **Memory, not speed, is the ship-blocker.** iOS kills on resident size. Check
  `os_proc_available_memory()`, request `com.apple.developer.kernel.increased-memory-limit`,
  and treat a 4B-class model on a phone as tight rather than routine.
- **Measuring throughput through a chat UI.** Numbers taken through SwiftUI are not comparable
  to anything. Use a headless entry point, warm the model first, and say which device and OS
  build produced the number.
- **Assuming a capability exists because the model is famous.** Check `kind` in the catalog.
  A chat model is not a VLM; an ASR model is not a diarizer.
- **Long single GPU batches on iOS.** A multi-minute uninterrupted GPU run gets killed. Chunk
  long work.
- **Thermals.** Sustained generation throttles within minutes; a short burst benchmark and a
  long-run benchmark disagree by tens of percent on the same phone.

## Verification — what is actually checked

Do not describe these models as "verified" without saying what was verified:

- Each bundle is gated against the original model before it is enrolled — the export is stepped
  against the fp32/fp16 reference on fixed inputs (token-exact for LLMs, `cos ≥ 0.999`
  otherwise), then re-gated after compression, then run on hardware.
- The gate strength **differs per model** and is stated on that model's card in the
  [model zoo](https://github.com/john-rocky/coreai-model-zoo). Do not flatten that into a blanket
  claim.
- The gates are run by the maintainer, not by an independent party. What makes them checkable is
  that the recipe (`models/<model>/recipe.toml`), the export script, and the verification script
  are all published — `python3 conversion/zoo_verify.py <hf-repo>` re-checks a published bundle's
  tokenizer, chat template, context length and declared precision against its source model
  without a GPU or a device.

If a user is deciding whether to ship on this, tell them that accurately rather than either
overselling it or waving them off.

## Not your call

Ask the human:

- Publishing anything to Hugging Face, or pushing to a namespace you were not handed.
- Bumping a catalog revision pin.
- Posting publicly about the package, or filing issues/PRs against `apple/*`.
- Reporting device numbers you did not measure on that device.

## Implementing the current plan

If you were handed this repository to build the designed-but-unbuilt work, start at
[`docs/HANDOFF.md`](docs/HANDOFF.md) — it indexes every design document, gives the build order,
and lists the preconditions that must be resolved on hardware before the code they gate is
written.

## More

- [README](README.md) — the ops table, the examples, the FoundationModels provider.
- [`docs/COOKBOOK.md`](docs/COOKBOOK.md) — every "I want to …" mapped to its snippet.
- [`Examples/`](Examples) — one buildable app per capability; start from the closest one.
- [model zoo](https://github.com/john-rocky/coreai-model-zoo) — the models, cards, recipes, and
  [`AGENTS.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/AGENTS.md) for porting a
  *new* model rather than consuming one.
- [awesome-core-ai](https://github.com/john-rocky/awesome-core-ai) — the wider ecosystem.
