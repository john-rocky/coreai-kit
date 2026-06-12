# Getting Started with CoreAIKit

Build an on-device LLM app on Apple's Core AI framework in about ten lines. No Python, no
model conversion — starter models are hosted on the Hugging Face Hub and download in-app.

## Requirements

- macOS 27 beta or iOS 27 beta (real device — the CoreAI framework is not in the iOS
  Simulator SDK), Xcode 27 beta
- ~1–5 GB of disk per model (cached after first download)

## 1. Add the package

In Xcode: File ▸ Add Package Dependencies… ▸ `https://github.com/john-rocky/coreai-kit`,
then add the `CoreAIKit` product to your target. Or in `Package.swift`:

```swift
.package(url: "https://github.com/john-rocky/coreai-kit", branch: "main"),
// target dependency:
.product(name: "CoreAIKit", package: "coreai-kit"),
```

## 2. Chat

```swift
import CoreAIKit

let chat = try await ChatSession(model: .qwen3_0_6B)   // downloads on first use
for try await event in chat.streamResponse(to: "What is the capital of Japan?") {
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
let chat = try await ChatSession(model: .qwen3_0_6B, configuration: config)
```

### Download progress

```swift
let chat = try await ChatSession(model: .qwen3_4B) { progress in
    print("\(progress.currentFile): \(Int(progress.fraction * 100))%")
}
```

Models cache under `Application Support/CoreAIKit/Models`. Manage them with `ModelStore`
(`downloadedModels()`, `delete(_:)`, or a custom `ModelStore(directory:)`).

## 3. Starter models

| Model | `ModelID` | macOS | iOS | Notes |
|---|---|---|---|---|
| Qwen3 0.6B | `.qwen3_0_6B` | ✓ | ✓ | smallest, thinking model |
| Qwen3 4B | `.qwen3_4B` | ✓ | ✓ | thinking model |
| Mistral 7B v0.3 | `.mistral_7B` | ✓ | – | |
| Gemma 3 4B | `.gemma3_4B` | ✓ | – | |

Any other Hugging Face repo with the same bundle layout works:
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

## 4. Use a local bundle

Already have a bundle exported with Apple's recipes (`coreai.llm.export`)?

```swift
let chat = try await ChatSession(bundleAt: URL(fileURLWithPath: "/path/to/bundle"))
```

The bundle directory holds `metadata.json`, a `*.aimodel/`, and a `tokenizer/`.

## 5. Run the example app

```bash
cd Examples/ChatDemo
xcodegen generate
open ChatDemo.xcodeproj
```

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

let model = try await KitLanguageModel(model: .qwen3_0_6B)
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

- The engine ignores a consumer break and keeps generating to
  `maximumResponseTokens` in the background; EOS-ended turns therefore reset the KV
  cache on the next turn (correctness first). Capped turns (no EOS) hit the append-only
  KV fast path and report `cachedTokenCount` in usage. Set `COREAI_KIT_DEBUG=1` to watch
  the decisions.
- Guided generation (`@Generable` structured output) is not implemented yet — schema
  requests throw `unsupportedCapability`.
- Usage/metadata events are sent once at end of turn (an upfront usage event
  materializes an empty transcript entry on tool turns in the current beta).

## Performance notes

- **Benchmark in Release.** The engine's per-token host work is ~3x slower in Debug builds.
- **First load compiles on-device** (one-time per model; cached afterwards). Subsequent
  loads are seconds.
- `stats.tokensPerSecond` is a 32-token rolling window — the engine bursts at decode
  start, so a cumulative average would over-read on short replies.
- Each turn re-prefills the full history (the robust official-runtime path). Long
  conversations grow TTFT; `reset()` clears it.

## Roadmap

- Tool dialects for more families (gpt-oss harmony, gemma, mistral)
- Guided generation (constrained decoding on engines that expose logits)
- More typed CV pipelines (depth, detection) over `GraphModel`
