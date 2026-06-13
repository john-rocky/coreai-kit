# SpotlightApp — ask your own files, with your own local model

A SwiftUI app (iPhone + Mac) that answers questions about **your real files** using **Apple's
WWDC26 `SpotlightSearchTool`** for retrieval — driven by **your own converted model**, not the
system one. Fully on device, works in airplane mode.

You point the app at a folder (or a set of files) with the picker; it searches them in place and
grounds every answer in the real text. The retrieval chain is the centerpiece of the UI: you watch
the model **search your files → read the ones that matter → answer**, and every answer carries the
source files it used (tap to open).

This is the file-backed evolution of [`Examples/SpotlightChat`](../SpotlightChat) (the verified
CLI): the same 2-tool RAG, now over your own documents, as an app you can run on a phone.

## What it shows

- `SpotlightSearchTool`'s **`.files` source** searches files you grant in place — no app index,
  no other-app access (that's sandboxed off by design). The corpus is whatever real folder/files
  you pick (security-scoped).
- It is a plain `FoundationModels.Tool`, so it rides behind **any** `LanguageModel`. Here it runs
  behind a Core AI zoo bundle via `KitLanguageModel` — local RAG on a third-party model, no Apple
  Intelligence required.
- The realistic 2-tool pattern: **`spotlight_search`** (Apple's tool) finds candidate files;
  **`fetch_file`** (this app's tool) reads a file's full text from disk; the model grounds its
  answer in what it actually read.
- The whole stack — model, provider, retrieval tool, your data — is on the device and open.

There is never an empty first screen: a few **sample documents** are seeded into the app's own
Application Support folder on first launch and used as the default scope, so you can ask a question
immediately, then switch to your own files.

## Run it

The project is generated with [XcodeGen](https://github.com/yonyz/XcodeGen):

```bash
cd Examples/SpotlightApp
xcodegen generate
open SpotlightApp.xcodeproj
```

- **Mac**: pick the **My Mac** destination and Run. Use **Choose ▸ Choose a folder…** to point it
  at, say, your `~/Documents` or a notes folder. (Scheme runs Release — the engine's per-token host
  work is ~3× slower unoptimized.)
- **iPhone / iPad**: connect the device, pick it as the destination, set your signing team on the
  `SpotlightApp` target, and Run. The folder/file picker grants access to whatever you select.
  iOS 27 / macOS 27.

On first launch the app downloads the default model (Qwen3 4B, ~2.5 GB) once, then it is cached —
every later launch, including the whole RAG round trip, runs with **no network**. Turn on airplane
mode and ask away.

### Headless self-test (the gate)

```bash
# from a built product dir, or via `xcodebuild ... build` then run the binary:
SPOTLIGHT_SELFTEST=1 /path/to/SpotlightApp.app/Contents/MacOS/SpotlightApp
SPOTLIGHT_SELFTEST=1 SPOTLIGHT_SELFTEST_ASK="Which file mentions a waterfall?" .../SpotlightApp
```

Seeds the sample documents, points the `.files` source at that folder, and runs the exact
`RagEngine` path the UI uses (search → fetch → grounded answer). It prints the retrieval chain and
the answer, and exits 0 only if the model actually read a file **and** produced an answer (i.e. it
grounded, not hallucinated).

## How it works

- **`RagEngine`** owns the loaded `KitLanguageModel` and runs one grounded answer per call over a
  `LibrarySnapshot` (the active files). Each question gets a **fresh `LanguageModelSession`** with
  the two tools — an independent retrieval over the corpus, so the prompt always fits the context
  window and the answer is always freshly searched (the UI still shows an accumulating
  conversation). Switching folders needs no model reload.
- **`FileLibrary`** resolves the picked roots into a snapshot (folders walked recursively, capped),
  reads file text (plain text / source via decoding, PDFs via PDFKit, RTF/HTML via
  `NSAttributedString`), and seeds the sample documents.
- The **live retrieval trace** comes from two side channels: `tool.searchResults` (an
  `AsyncSequence` of what the search surfaced) and a reporter injected into `FetchFileTool` (which
  files were read → the citations). The model's own search query is recovered from the finished
  `session.transcript` for the trace. The answer **streams** via `session.streamResponse(to:)`.

## Things that matter (carried from the CLI verification)

- **The search tool returns metadata, not your content.** `SpotlightSearchTool` hands the model
  names / paths / dates — never the body. That's why the second `fetch_file` tool exists; without
  it a model "answers from search" by hallucinating bodies from filenames.
- **Guidance is a token budget.** The default `.complete` guidance is ~13k tokens and overflows a
  4k-context model on contact. This app ships `.focused(.items)` + `format: .compact`. A fetched
  file's text is also truncated to one context window's worth.
- **The default model is Qwen3 4B.** Tool calling through the kit needs a ChatML tokenizer (the
  qwen3 family); the 0.6B is too small for this rich tool schema (it loops). The instructions end
  with **`/no_think`** to disable qwen3's chain-of-thought — otherwise the reasoning can run to the
  token cap and the framework reports "ended without producing a response." Harmless to non-qwen
  models.

## Files

| File | Role |
|---|---|
| `FileLibrary.swift` | The user's real files: scopes, recursive walk, text extraction, sample seeding |
| `Tools.swift` | `FetchFileTool` (hydration) + live `SearchReply` → UI mapping |
| `RagEngine.swift` | The verified 2-tool RAG, headless — loads the model, answers one question over a library |
| `ChatModel.swift` | `@Observable` view model: library selection, loading, live callbacks → UI state |
| `ContentView.swift` | The chat UI: identity + badges, library bar, the visible RAG trace, tappable sources |
| `SelfTest.swift` | `SPOTLIGHT_SELFTEST=1` headless gate (over the sample folder) |

A real app's bundle identity makes the Foundation Models tool path work out of the box — unlike the
CLI, no embedded `Info.plist` hack is needed.
