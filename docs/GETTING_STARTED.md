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

Any other stateless `.aimodel` graph runs through the generic `GraphModel`:

```swift
let model = try await GraphModel(contentsOf: aimodelURL, computeUnits: .neuralEngine)
let out = try await model.run(["pixel_values": .float32(pixels, shape: [1, 3, 224, 224])])
let depth = out["depth"]!.floats()
```

## Performance notes

- **Benchmark in Release.** The engine's per-token host work is ~3x slower in Debug builds.
- **First load compiles on-device** (one-time per model; cached afterwards). Subsequent
  loads are seconds.
- `stats.tokensPerSecond` is a 32-token rolling window — the engine bursts at decode
  start, so a cumulative average would over-read on short replies.
- Each turn re-prefills the full history (the robust official-runtime path). Long
  conversations grow TTFT; `reset()` clears it.

## Coming next

- `CoreAIKitVision`: run CLIP and friends in a few lines (`Examples/PhotoSearch`)
- `KitLanguageModel`: your models behind Apple's `LanguageModelSession`, with tool calling
