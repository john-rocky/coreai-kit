# SpotlightApp — ask your own notes, with your own local model

A SwiftUI chat app (iPhone + Mac) that answers questions about your notes using **Apple's
WWDC26 `SpotlightSearchTool`** for retrieval — driven by **your own converted model**, not the
system one. Fully on device, works in airplane mode. The retrieval chain is visible: you watch
the model search and read your notes, and every answer carries the source notes it was grounded
in (tap to read the full text).

This is the SwiftUI version of [`Examples/SpotlightChat`](../SpotlightChat) (the verified CLI).
Same 2-tool RAG, now an app you can run on a phone.

## What it shows

- `SpotlightSearchTool` is a plain `FoundationModels.Tool`, so it rides behind **any**
  `LanguageModel`. Here it runs behind a Core AI zoo bundle via `KitLanguageModel` — local RAG
  on a third-party model, no Apple Intelligence required.
- The realistic 2-tool pattern: **`spotlight_search`** (Apple's tool) finds candidate notes in
  the Spotlight index; **`fetch_note`** (this app's tool) reads a note's full body from the app's
  own store; the model grounds its answer in the text it actually read.
- The whole stack — model, provider, retrieval tool, your data — is on the device and open.

## Run it

The project is generated with [XcodeGen](https://github.com/yonyz/XcodeGen):

```bash
cd Examples/SpotlightApp
xcodegen generate
open SpotlightApp.xcodeproj
```

- **Mac**: pick the **My Mac** destination and Run. (Scheme runs Release — the engine's
  per-token host work is ~3× slower unoptimized.)
- **iPhone / iPad**: connect the device, pick it as the destination, set your signing team on the
  `SpotlightApp` target (it ships with a placeholder), and Run. iOS 27 / macOS 27.

On first launch the app downloads the default model (Qwen3 4B, ~2.5 GB) once, then it is cached —
every later launch, including the whole RAG round trip, runs with **no network**. Turn on airplane
mode and ask away.

### Headless self-test (the gate)

```bash
# from a built product dir, or via `xcodebuild ... build` then run the binary:
SPOTLIGHT_SELFTEST=1 /path/to/SpotlightApp.app/Contents/MacOS/SpotlightApp
SPOTLIGHT_SELFTEST=1 SPOTLIGHT_SELFTEST_ASK="Where did I see eagles?" .../SpotlightApp
```

Runs the exact `RagEngine` path the UI uses (index → search → fetch → grounded answer), prints
the retrieval chain and the answer, and exits 0 only if the model actually read a note **and**
produced an answer (i.e. it grounded, not hallucinated).

## How it works

- **`RagEngine`** owns the loaded `KitLanguageModel` and runs one grounded answer per call. Each
  question gets a **fresh `LanguageModelSession`** with the two tools — an independent retrieval
  over the corpus, so the prompt always fits the 4k window and the answer is always freshly
  searched (the UI still shows an accumulating conversation).
- The **live retrieval trace** comes from two side channels: `tool.searchResults` (an
  `AsyncSequence` of what Spotlight surfaced) and a reporter injected into `FetchNoteTool` (which
  bodies were read → the citations). The model's own search query is recovered from the finished
  `session.transcript` for the "How it answered" view.
- The answer **streams** via `session.streamResponse(to:)`.

## Things that matter (carried from the CLI verification)

- **The search tool returns metadata, not your content.** `SpotlightSearchTool` hands the model
  titles / ids / dates — never the body, even when you index `contentDescription`. That's why the
  second `fetch_note` tool exists; without it a model "answers from search" by hallucinating
  bodies from titles.
- **Guidance is a token budget.** The default `.complete` guidance is ~13k tokens and overflows a
  4k-context model on contact. This app ships `.focused(.items)` + `format: .compact`.
- **The default model is Qwen3 4B.** Tool calling through the kit needs a ChatML tokenizer (the
  qwen3 family); the 0.6B is too small for this rich tool schema (it loops). The instructions end
  with **`/no_think`** to disable qwen3's chain-of-thought — with this big tool schema the
  reasoning can otherwise run to the token cap and the framework reports "ended without producing
  a response." Harmless to non-qwen models.

## Files

| File | Role |
|---|---|
| `Notes.swift` | The five sample notes + Core Spotlight indexing + index delegate |
| `Tools.swift` | `FetchNoteTool` (hydration) + live `SearchReply` → UI mapping |
| `RagEngine.swift` | The verified 2-tool RAG, headless — loads the model, answers one question |
| `ChatModel.swift` | `@Observable` view model: indexing, loading, live callbacks → UI state |
| `ContentView.swift` | The chat UI: retrieval trace, streamed answer, tappable source notes |
| `SelfTest.swift` | `SPOTLIGHT_SELFTEST=1` headless gate |

A real app's bundle identity makes `CSSearchableIndex.default()` work out of the box — unlike the
CLI, no embedded `Info.plist` hack is needed.
