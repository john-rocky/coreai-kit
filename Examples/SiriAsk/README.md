# Gemma (SiriAsk) — "Hey Siri, Ask Gemma" → your on-device Gemma 4 answers anything

Say **"Hey Siri, Ask Gemma"** (or tap **Ask** in the app) and a **Gemma 4 E2B** model running
entirely **on your iPhone** answers free-text questions from its own knowledge. No notes, no
retrieval, no tools — just ask-anything Q&A. It works in airplane mode and nothing leaves the phone.

The point vs "just use Apple Intelligence": **you choose the model.** Siri does the routing and the
talking; the *answering* is a Core AI zoo bundle behind FoundationModels' `LanguageModelSession`,
via the kit's `KitGemmaModel` (GEMMA-KIT). The app's display name is **Gemma**, so the Siri phrase
`Ask \(.applicationName)` speaks as **"Ask Gemma"**.

> This is the ask-anything rework of the earlier notes-RAG SiriAsk. Dropping retrieval also drops
> the whole prompt-injection surface (see `SECURITY.md`): with no untrusted data channel and no
> tool, the Lethal Trifecta has nothing to actuate.

## What it shows

- **A third-party model behind Siri / App Intents.** `AskGemmaIntent.perform()` runs
  `LanguageModelSession(model: KitGemmaModel(.gemma4E2B…))`. Siri routes to the intent; the intent
  runs *your* model — Siri's own model never hosts a 3rd-party LM (that's why this works at all).
- **Gemma 4's PLE behind FoundationModels.** Gemma 4 small models carry a giant per-layer-embedding
  table; the published `…_tbl` bundle exposes it as two **static graph inputs** the runtime binds
  once. GEMMA-KIT's `KitGemmaModel` / `GemmaRuntime` wires that behind the `LanguageModel` protocol.
- **On-device, the hard parts handled:** the AOT-compiled iPhone bundle, the increased-memory
  entitlement, and warm residency so Siri's tight intent budget reuses an already-loaded model.

## The iPhone reality (why this example exists)

Three things make on-device Gemma 4 non-trivial, all handled here:

1. **AOT bundle.** On device the plain `…_tbl` bundle's ~2 GB of graph constants crash the on-device
   GPU specializer (`LLVM ERROR: Failed to allocate mmap'd buffer`). The iPhone build must load the
   **AOT-compiled** `gemma4_e2b_qat_decode_int4lin_tbl_aotc_h18p` bundle (h18p = iPhone 17 Pro GPU
   family). `GemmaSource` (in `ModelAsk.swift`) selects it on iOS and the plain `…_tbl` bundle on
   macOS (which JIT-specializes fine). Both pair with the same QAT PLE tables.
2. **Increased-memory entitlement.** Gemma 4 E2B peaks ~4.4 GB resident (decode bundle + ~2.4 GB PLE
   tables); the default jetsam limit SIGKILLs that. `SiriAsk.entitlements` carries
   `com.apple.developer.kernel.increased-memory-limit` (~6.44 GB on iPhone 17 Pro).
3. **Warm residency.** A multi-GB model can't cold-load inside Siri's intent budget, so the app
   warms `ModelHost.shared` on launch and reuses it from the intent in the same process. The first
   answer after launch is slow (documented, not hidden).

The raw inference is device-proven: `ondevice/_gemma4_qat_RESULTS.md` measured this exact AOT bundle
on an iPhone 17 Pro at decode ~30 tok/s, oracle 8/8 (token-identical to the Mac), ~4.45 GB peak.

## Design: tool-less, no retrieval

```
Siri → AskGemmaIntent.perform()
        → ModelHost.shared.ask(question:)               // warm, serial (actor)
        → LanguageModelSession(model: KitGemmaModel).respond   // tool-less ask-anything
        → IntentDialog(answer)                           // Siri speaks it
```

No retrieval to ground, no tool to call, no destructive intent exposed. The model only ever reads
the question and answers.

## Layout

```
SiriAsk/
  Package.swift              # SiriAskCore (lib) + SiriAskGate (exe) + tests — runs on a Mac
  Sources/SiriAskCore/
    ModelAsk.swift           #   GemmaSource (per-platform bundle) + ModelHost actor (warm ask)
  Sources/SiriAskGate/       #   headless Mac sanity (loads Gemma, asks free-text questions)
  Tests/SiriAskCoreTests/    #   ModelSourceTests — asserts iOS=AOT bundle / macOS=plain (no GPU)
  App/                       # the Siri / App Intents / SwiftUI app (XcodeGen)
    project.yml              #   iOS + macOS, deployment 27, Gemma display name, entitlement
    SiriAsk.entitlements     #   increased-memory-limit
    Sources/
      SiriAskApp.swift       #     @main; ContentView warms the model on appear
      Intents.swift          #     AskGemmaIntent (free-text, read-only)
      AppShortcuts.swift     #     "Ask Gemma" phrases
      ContentView.swift      #     coach home: in-app Ask box + Siri phrase card
      PhraseCard.swift       #     AskBox (the in-app G1 surface) + SiriPhraseCard
      DemoModels.swift       #     ModelStatus (warm-on-launch) + AskRun
      DemoShell.swift        #     shared demo shell (identity header, on-device/offline badge)
```

## Run it

Headless core on a Mac (no device):

```bash
cd Examples/SiriAsk
swift test                                    # model-source invariant (iOS AOT / macOS plain), no GPU
# Optional model sanity (loads Gemma 4 on the GPU — hold the repo _GPU_LOCK):
swift run -c release SiriAskGate              # downloads the macOS Gemma 4 E2B bundle, asks 3 questions
swift run -c release SiriAskGate --decoder <dir> --tables <dir>   # or point at local bundle dirs
```

The app (Siri + the on-device model need a real build):

```bash
cd Examples/SiriAsk/App
xcodegen generate
open SiriAsk.xcodeproj      # run on iPhone 17 Pro; first launch downloads ~4.4 GB, then warms
```

- In-app: type a question (or tap a sample) → **Ask Gemma** → the on-device answer + timing.
- With Siri (app run once so the model is warm): **"Hey Siri, Ask Gemma"** → Siri asks "What do you
  want to ask Gemma?" → say your question → Gemma answers out loud.

## Notes / known constraints

- **Model = Gemma 4 E2B, official QAT int4** (license: gemma). iPhone path = the AOT
  `…_tbl_aotc_h18p` bundle; Mac path = the plain `…_tbl` bundle. E4B is Mac-only (the static-table
  form exceeds the iPhone memory budget). Both from `mlboydaisuke/gemma-4-E2B-CoreAI`.
- **Free-text in shortcut phrases isn't allowed.** App Intents only lets `AppEntity`/`AppEnum`
  parameters be interpolated into an `AppShortcut` phrase — not a `String`. So "Ask Gemma" is the
  trigger and Siri prompts for the question (`requestValueDialog`). Two steps, by design.
- **Latency.** A multi-GB model can't cold-load inside Siri's intent budget; the app keeps it warm
  (`ModelHost.shared`, prewarmed on launch). First call after a cold launch is slow — documented.
- Built + verified on macOS 27 beta (`swift build`/`swift test`, macOS app `BUILD SUCCEEDED` with
  the `AskGemmaIntent` + "Ask Gemma" shortcut metadata, iOS-device slice compiles). The on-device
  load + answer (and the Siri voice flow) are verified on iPhone 17 Pro hardware (USER).
