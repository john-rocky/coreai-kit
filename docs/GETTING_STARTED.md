# Getting Started with CoreAIKit

Build an on-device LLM app on Apple's Core AI framework in about ten lines. No Python, no
model conversion — starter models are hosted on the Hugging Face Hub and download in-app.

## Requirements

- macOS 27 beta or iOS 27 beta (real device — the CoreAI framework is not in the iOS
  Simulator SDK), Xcode 27 beta
- Qwen3 0.6B starter: approximately 352 MB on Mac / 456 MB on iPhone; allow at least
  1 GB of free disk. Other capabilities download their own models.

## Release 0.4.1 validation

The release entry is **Qwen3 0.6B**, from
[`mlboydaisuke/qwen3-0.6b-CoreAI-official`](https://huggingface.co/mlboydaisuke/qwen3-0.6b-CoreAI-official/tree/943eb6a4f967de53d7e1458d75deac0b68ac3d85),
revision `943eb6a4f967de53d7e1458d75deac0b68ac3d85`. Its pinned `macos` files total
**351,561,081 bytes**, and the `ios` files total **456,296,751 bytes**. These are download
bytes, not peak memory. The package selects the platform bundle and caches it.

The package is tested on **Mac Studio M4 Max (128 GiB)**, **macOS 27.0 beta `26A5416b`**,
**Xcode 27 beta 5 `27A5237l`**, **macOS SDK `26A5406c`**. Its public runtime is
**coreai-models `0.2.4-zoo`**, revision `f7a75ec0f89fab451d277572afe8995b7ef768c1`.
The independent public consumer resolved **swift-transformers `1.3.4`**, revision
`c21fdcde390313a6d98d8e33a346f2c3486c3ab0`; retain your consumer's `Package.resolved`.

The independent consumer resolved the **published exact `0.4.1` tag**, commit
`11f2823b9ba9c059b4db86c4e1682c53112006f1`, before its empty-cache download,
two-turn FoundationModels check, and both Japanese checks below. The
[public evidence ZIP](https://github.com/john-rocky/coreai-kit/releases/download/0.4.1/coreaikit-0.4.1-macos-evidence-20260909.zip)
contains the consumer's source, manifest, lockfile, actual output, and model-file
hashes. Its [SHA-256](https://github.com/john-rocky/coreai-kit/releases/download/0.4.1/coreaikit-0.4.1-macos-evidence-20260909.zip.sha256)
and the archive's per-file checksums make those artifacts checkable.

| Check | Observed result on the Mac |
|---|---|
| First-use Qwen download → answer | Empty dedicated cache; all 11 files match the public revision's size and hash. Nonempty answer; cache reuse passes. |
| FoundationModels, two turns | First reply confirms ORCHID; second recalls ORCHID. |
| Japanese, ChatSession | 177 characters, six deltas, final message matches the assembled stream. |
| Japanese, FoundationModels | 177 characters, 116 cumulative snapshots, no U+FFFD replacement characters. |
| Hybrid Qwen3.5 0.8B | Both turns complete without a partial-reset error; turn two recalls ORCHID. The first reply refuses the instruction, so this is an engine regression check, not an answer-quality endorsement. |
| VoxCPM 0.5B | New synthesis from verified public cached weights: 20,480 finite, nonzero samples; 1.28 s, 16 kHz mono WAV. Playback exits successfully. |

The hybrid and VoxCPM rows reuse the preceding public candidate `61f7c454` runs.
The archive separates them from the exact-tag checks and includes the full library
diff: only the starter's iOS size metadata changed; Mac implementation, runtime and
model pins stayed the same. The [0.4.1 Mac demo and reproduction record](https://github.com/john-rocky/coreai-assets/blob/main/kit/coreaikit-0.4.1-mac.md)
add an exact-tag ChatDemo → Speak run on the same physical Mac, with first-use
Speak downloads completed before recording the cached run.

Qwen3.5 0.8B uses revision `1b8c0203c0f317027db508e97372c589afaace7b`;
its selected files total **1,337,858,033 bytes**, downloaded and hash-checked against
the public revision.

The Japanese prompt requests around 400 characters; the model returns a shorter,
imperfect answer. This verifies the streaming path, not translation quality. VoxCPM
uses revision `9920f9599f98ad257f09cd606240d79282db3991`, with `macos`, `tokenizer`
and `voxcpm_host_glue` totaling **1,373,894,365 bytes**. Those files were hash-checked
against the public revision and reused in a dedicated cache; this was a new synthesis,
not a claim of a fresh TTS download or subjective voice-quality approval.

The [ChatDemo release checks](../Examples/ChatDemo#reproduce-the-release-checks)
provide the executable source and commands for an empty dedicated model cache,
FoundationModels two-turn recall, and Japanese streaming. They are maintainer-run
checks, not independent adoption or a guarantee of a small model's answer quality.
The [release](https://github.com/john-rocky/coreai-kit/releases/tag/0.4.1) holds the
final result and public consumer evidence.

**iPhone execution, GA and newer beta toolchains are not established by these results.** As observed on
September 9, the [Apple release list](https://developer.apple.com/news/releases/) still
lists OS 27 beta 8 and Xcode 27 beta 6; this entry uses the beta 5 SDK generation.
Do not update a shared machine's OS/SDK merely to reproduce the demo.

## 1. Add the package

In Xcode: File ▸ Add Package Dependencies… ▸ `https://github.com/john-rocky/coreai-kit`,
then add the `CoreAIKit` product to your target. Or in `Package.swift`:

```swift
.package(url: "https://github.com/john-rocky/coreai-kit", exact: "0.4.1"),
// target dependency:
.product(name: "CoreAIKit", package: "coreai-kit"),
```

## 2. The quick path: task ops

`CoreAIOps` (a separate product in the same package) is the task-level layer: state the
task and the kit resolves — and caches — a catalog model behind it, like a Vision
framework request. Every op takes `options: .model("catalog-id")` to override the model
per call, and because the ops ride the model-level APIs below, outgrowing one is a
refactor, not a rewrite. Adding this one product is enough — `import CoreAIOps`
re-exports the model layer (`ChatSession`, `KitDetector`, `TextEmbedder`, …), so no
snippet in this guide needs a second product or import.

```swift
import CoreAIOps

let text = try await CoreAI.transcribe(voiceMemoURL)            // speech → text
let tldr = try await CoreAI.summarize(text, style: .bullets)    // text → summary
let todo = try await CoreAI.extract(text, as: ActionItems.self) // text → @Generable type
let ja   = try await CoreAI.translate(text, to: .japanese)
let safe = try await CoreAI.redact(text)                        // "[PERSON]", "[EMAIL]", …

let cap  = try await CoreAI.caption(photo)                      // image → description
let dets = try await CoreAI.detect(in: photo)                   // image → [Detection]
let page = try await CoreAI.read(documentAt: scanURL)           // document → markdown
let hits = try await CoreAI.search(query, in: paragraphs)       // semantic ranking
let wave = try await CoreAI.speak(reply)                        // text → speech (PCM)
```

Twenty-four ops in all — also `proofread`, `tidyTranscript`, `extractEntities`, `transcribeMeeting` (who said
what), `describeAudio`, `compose` (text → music), `separate` (vocals / instrumental),
`upscale`, `estimateDepth`, `recognizeAction`, and `forecast` (time series). The
[Cookbook](COOKBOOK.md) maps every "I want to …" to its snippet; `Examples/OpsDemo`
runs the core pipeline end to end.

First use of an op downloads its model (cached afterwards). Watch and front-load that
from one place — no progress parameter on two dozen ops:

```swift
CoreAI.onDownload { print("\($0.currentFile): \(Int($0.fraction * 100))%") }
try await CoreAI.prepare(.transcribe, .summarize)   // behind your loading UI;
                                                    // the first real call starts instantly
```

The rest of this guide is the model-level layer: you pick the model, hold the session,
and stream.

## 3. Chat

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "qwen3-0.6b")   // downloads on first use
for try await event in await chat.streamResponse(to: "What is the capital of Japan?") {
    switch event {
    case .response(let delta): print(delta, terminator: "")
    case .thinking(let delta): break   // qwen3 reasoning, if you want to show it
    case .stats(let stats):    break   // live TTFT / tok/s / token counts
    case .complete(let message): print("\nDone: \(message.content.count) chars")
    }
}
```

One-shot convenience without events:

```swift
let answer = try await chat.respond(to: "And its population?")  // history carries over
```

`ChatSession` keeps the conversation history (`chat.history`), re-rendering it through the
bundle's own chat template every turn. `chat.reset()` starts a fresh conversation without
reloading the model; `chat.cancelGeneration()` is your stop button (the stream completes
with the partial message).

### Configuration

```swift
var config = ChatSession.Configuration()
config.temperature = nil            // greedy decoding (default 0.7)
config.maxResponseTokens = 1024     // default 2048
config.systemPrompt = "You are a terse assistant."
let chat = try await ChatSession(catalog: "qwen3-0.6b", configuration: config)
```

### Download progress

```swift
let chat = try await ChatSession(model: .qwen3_4B) { progress in
    print("\(progress.currentFile): \(Int(progress.fraction * 100))%")
}
```

Models cache under `Application Support/CoreAIKit/Models`. Manage them with `ModelStore`
(`downloadedModels()`, `delete(_:)`, or a custom `ModelStore(directory:)`).

### HF-compatible endpoints (0.4.1+)

Pass a store configured for your chosen HF-compatible endpoint. Both the tree listing
and file downloads use that URL, including an optional path prefix:

```swift
let store = ModelStore(hubBaseURL: URL(string: "https://your-mirror.example/hf")!)
let chat = try await ChatSession(catalog: "qwen3-0.6b", store: store)
```

Replace the example URL with your endpoint. `ModelStore(directory:hubBaseURL:)` also
accepts a custom cache directory. The default stays `https://huggingface.co`; catalog
revision pins and the repo/revision/variant cache layout are preserved across endpoints.
This setting does not change the catalog's GitHub URL. The kit does not read `HF_TOKEN`
or copy HF authentication headers to a mirror; base URLs containing credentials, a query
or a fragment are rejected. Private-repo authentication and third-party mirror compatibility
have not been validated by this change.

## 4. Starter models

| Model | Catalog ID | macOS | iOS | Notes |
|---|---|---|---|---|
| Qwen3 0.6B | `qwen3-0.6b` | ✓ | ✓ | smallest, thinking model |
| Qwen3 4B | `qwen3-4b` | ✓ | ✓ | thinking model |
| Mistral 7B v0.3 | `mistral-7b-v0.3` | ✓ | – | |
| Gemma 3 4B | `gemma-3-4b-it` | ✓ | – | |

Use these IDs with `ChatSession(catalog:)`. Catalog initializers resolve the current
reviewed pin; for an immutable release snapshot, resolve `ModelCatalog.builtin` as in
the FoundationModels example. Legacy static presets such as `.qwen3_0_6B` use the
model repo’s `main` branch and are not the reproducible release entry.

A compatible Hugging Face repo with the same bundle layout can be specified directly:
`ModelID("org/name", path: "macos")`.

The live list (with download sizes) is also available as a remote catalog — new models
reach your picker without a package update:

```swift
let catalog = await ModelCatalog.load()        // falls back to a built-in snapshot offline
for entry in catalog.available(.chat) {
    print(entry.name, entry.variant?.sizeMB ?? 0, "MB")   // entry.modelID feeds ChatSession
}
```

Tip: call `try await chat.prewarm()` right after init (while your UI still shows a
loading state) — it compiles the sampler graph so the first turn starts instantly.

## 5. Use a local bundle

Already have a bundle exported with Apple's recipes (`coreai.llm.export`)?

```swift
let chat = try await ChatSession(bundleAt: URL(fileURLWithPath: "/path/to/bundle"))
```

The bundle directory holds `metadata.json`, a `*.aimodel/`, and a `tokenizer/`.

## 6. Run the example app

```bash
cd Examples/ChatDemo
xcodegen generate          # brew install xcodegen (once)
open ChatDemo.xcodeproj
```

Signing: the example projects don't hard-code a team. Either pick yours once in
Xcode's Signing & Capabilities tab, or `export DEVELOPMENT_TEAM=XXXXXXXXXX` (your
team id) before `xcodegen generate` and every example picks it up.

## Computer vision: CLIP in five lines

`CoreAIKitVision` is a separate product — CV apps don't link any LLM runtime.

```swift
import CoreAIKitVision

let encoder = try await ImageTextEncoder()   // downloads CLIP ViT-B/32 (~290 MB) on first use
let imageVec = try await encoder.encode(image: cgImage)        // preprocessing included
let textVec  = try await encoder.encode(text: "red bike at the beach")
let score = ImageTextEncoder.cosineSimilarity(imageVec, textVec)
```

Embeddings are L2-normalized 512-d vectors; ranking a photo library is one dot product per
photo (see `Examples/PhotoSearch`).

Monocular depth is two lines (`Examples/DepthCamera` runs it live):

```swift
let depth = try await DepthEstimator()       // downloads Depth Anything 3 small (~100 MB)
let map = try await depth.estimateDepth(for: cgImage)
imageView.image = map.cgImage()              // min-max-normalized grayscale
```

Live camera pipelines are a for-await loop:

```swift
for await frame in try await CameraFeed(framesPerSecond: 5).start() {
    let map = try await depth.estimateDepth(for: frame)
}
// The app needs NSCameraUsageDescription; only the newest frame is buffered, so slow
// consumers skip frames instead of lagging.
```

Any other stateless `.aimodel` graph runs through the generic `GraphModel`:

```swift
let model = try await GraphModel(contentsOf: aimodelURL, computeUnits: .neuralEngine)
let out = try await model.run(["pixel_values": .float32(pixels, shape: [1, 3, 224, 224])])
let depth = out["depth"]!.floats()
```

## SwiftUI components

`CoreAIKitUI` ships the pieces every model app rebuilds — a catalog-driven
`ModelPickerBar` (selection + load button + status + download progress), a
`ChatTranscriptView` (bubbles, thinking disclosure, auto-scroll), and a `StatsBar`
(load / TTFT / tok/s / memory). `Examples/ChatDemo` is built from them; its whole UI
fits in ~60 lines.

## Text embeddings and local RAG

`CoreAIKitEmbeddings` gives you on-device text embeddings (EmbeddingGemma, normalized
768-d, multilingual):

```swift
import CoreAIKitEmbeddings

let embedder = try await TextEmbedder()        // downloads EmbeddingGemma (~590 MB)
let doc = try await embedder.embed(document: "Tokyo is the capital of Japan.")
let query = try await embedder.embed(query: "what is the capital of Japan")
let score = TextEmbedder.cosineSimilarity(doc, query)
```

The asymmetric retrieval prompts (query vs document) are applied automatically. Combine
with `KitLanguageModel` and a retrieval `Tool` for complete on-device RAG — the model
decides when to search, the framework executes it, and the answer is grounded on your
documents. `Examples/DocChat` is the whole loop in ~150 lines:

```bash
swift run -c release DocChat ~/notes "What do my notes say about the bike trip?"
```

## Tool calling with LanguageModelSession

`KitLanguageModel` puts a Core AI bundle behind Apple's FoundationModels session API —
including tool calling, which Apple's own `CoreAILanguageModel` adapter does not
implement.

```swift
import CoreAIKit
import FoundationModels

struct WeatherTool: Tool {
    let name = "get_weather"
    let description = "Get the current weather for a city."

    @Generable
    struct Arguments {
        @Guide(description: "Name of the city, in English")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        "Sunny, 24 degrees Celsius in \(arguments.city)."
    }
}

guard let modelID = ModelCatalog.builtin.entry(id: "qwen3-0.6b")?.modelID else {
    throw CoreAIKitError.modelNotAvailableOnPlatform(id: "qwen3-0.6b")
}
let model = try await KitLanguageModel(model: modelID)
let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let answer = try await session.respond(to: "What's the weather in Sapporo right now?")
```

The framework owns the conversation transcript (persist `session.transcript` and restore
it to continue a conversation later), executes tool calls, and replays results into the
next model turn. Retrieval-augmented flows are "define a retrieval tool" — the model
decides when to search.

### Support matrix

| Model | ChatSession | thinking | FM chat | FM tools |
|---|---|---|---|---|
| Qwen3 0.6B / 4B | ✓ | `<think>` | ✓ | ✓ (Hermes ChatML) |
| Mistral 7B v0.3 | ✓ | – | ✓ | not yet (dialect) |
| Gemma 3 4B | ✓ | – | ✓ | not yet (dialect) |
| gpt-oss (local bundle) | ✓ (harmony parsed) | analysis channel | – | not yet |

### Known beta caveats

- The pinned `0.2.4-zoo` engine stops promptly after the consumer stops. The executor
  settles the previous generation and rewinds to the shared token prefix on the next
  turn. Hybrid engines that cannot partially rewind fall back to full re-prefill.
  Set `COREAI_KIT_DEBUG=1` to inspect cache decisions.
- Guided generation needs the sequential engine (per-step logits); on the default
  pipelined engine schema requests throw `unsupportedCapability`.
- Usage/metadata events are sent once at end of turn (an upfront usage event
  materializes an empty transcript entry on tool turns in the current beta).

## Guided generation (schema-valid JSON by construction)

Constrained decoding masks the model's per-step logits through a JSON schema's grammar
(xgrammar bitmask) before sampling — the output **cannot** violate the schema, so there
are no parse retries. It needs per-step logits, which the default GPU-pipelined engine
does not expose: load with `engineVariant: .sequential` (slower decode; fine for short
structured output). Constrained turns can't think — the grammar starts at the JSON.

With `ChatSession` (JSON schema string in, `Codable` out):

```swift
struct CityFacts: Codable { let name: String; let country: String }

var config = ChatSession.Configuration()
config.engineVariant = .sequential
let chat = try await ChatSession(catalog: "qwen3-0.6b", configuration: config)

let facts = try await chat.respond(
    to: "Give facts about the capital of Japan.",
    generating: CityFacts.self,
    schema: """
        {"type": "object",
         "properties": {"name": {"type": "string"}, "country": {"type": "string"}},
         "required": ["name", "country"]}
        """)
```

`respondJSON(to:schema:)` returns the raw JSON text, and `streamGuidedResponse` streams
it token by token. With FoundationModels, `@Generable` types work end to end:

```swift
guard let modelID = ModelCatalog.builtin.entry(id: "qwen3-0.6b")?.modelID else {
    throw CoreAIKitError.modelNotAvailableOnPlatform(id: "qwen3-0.6b")
}
let model = try await KitLanguageModel(model: modelID, engineVariant: .sequential)
let session = LanguageModelSession(model: model)
let plan = try await session.respond(to: "Plan a trip.", generating: TravelPlan.self)
```

See `Examples/GuidedDemo` for both paths runnable from the command line.

## Performance notes

- **Benchmark in Release.** The engine's per-token host work is ~3x slower in Debug builds.
- **First load compiles on-device** (one-time per model; cached afterwards). Subsequent
  loads are seconds.
- `stats.tokensPerSecond` is a 32-token rolling window — the engine bursts at decode
  start, so a cumulative average would over-read on short replies.
- The engine keeps its KV cache across turns and prefills only the tokens beyond the
  longest shared prefix with the previous turn (implicit prefix caching + `reset(to:)`
  rewind), so multi-turn TTFT stays flat. Engines that can't rewind mid-sequence
  (recurrent/SSM hybrids) fall back to a full re-prefill on divergence — lossless
  either way. `reset()` clears the conversation and the cache.
- The first turn whose prefill length exceeds every earlier one pays a one-time engine
  cost at that length (shape specialization — a 2-4 s TTFT spike on device). On
  dynamic-shape (GPU) engines, warm it away during loading with
  `prewarm(prefillLength: 256)` (any length ≥ your typical prompts; covers everything
  shorter). Static-shape (ANE) engines skip the warm internally, and it is wasted
  per-token work on S=1 zoo ports (catalog hint "pipelined") — leave it nil there.

## Roadmap

- Tool dialects for more families (gpt-oss harmony, gemma, mistral)
- Guided generation on the pipelined engine (needs an engine logits path)
- More typed CV pipelines (depth, detection) over `GraphModel`
