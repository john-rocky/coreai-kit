# Two layers, one package

**Task ops** when you want the result in one line — like a Vision framework request, the
model is resolved (and cached) behind the op:

```swift
import CoreAIOps

let text  = try await CoreAI.transcribe(voiceMemoURL)   // speech → text (Apple's, 0 bytes)
let tldr  = try await CoreAI.summarize(text)            // also: extract / translate / redact …
let boxes = try await CoreAI.detect(in: photo)          // [Detection] — RF-DETR, no NMS
let reply = try await CoreAI.speak(tldr)                // text → speech (PCM + sample rate)
```

Twenty-four ops, one shape — the [Cookbook](COOKBOOK.md) maps every "I want to …" to
its snippet. Adding the one `CoreAIOps` product is enough: it re-exports the model
layer, so the `import` above also covers everything below. First-use downloads are
observable process-wide (`CoreAI.onDownload { … }`) and prefetchable behind a loading
UI (`try await CoreAI.prepare(.transcribe, .caption)`) — and answerable before you offer
the feature at all:

```swift
switch await CoreAI.capability(.transcribeMeeting) {
case .ready:                     showButton()          // nothing to fetch
case .needsDownload(let bytes):  showPrompt(bytes)     // "Meeting notes needs 238 MB"
case .needsSystemAssets:         showFirstRunNotice()  // the OS's bytes, not the app's
case .insufficientStorage, .unsupportedDevice: hideFeature()
}
```

`swift run coreai-doctor path/to/YourApp` totals it for a whole app before you ship.

**Model-level APIs** when you want control — `ChatSession` (the quickstart above),
`KitLanguageModel` / `KitVisionModel` behind `LanguageModelSession`, `GraphModel` for any
`.aimodel`. Both layers are the same package, so starting with an op and dropping down
later is a refactor, not a rewrite.
