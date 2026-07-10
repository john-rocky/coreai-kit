# CoreAIKit

[![CI](https://github.com/john-rocky/coreai-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/john-rocky/coreai-kit/actions/workflows/ci.yml)
[![Nightly gate](https://github.com/john-rocky/coreai-kit/actions/workflows/nightly-gate.yml/badge.svg)](https://github.com/john-rocky/coreai-kit/actions/workflows/nightly-gate.yml)
[![Next-SDK models](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjohn-rocky%2Fcoreai-kit%2Fgate-status%2Fnext-sdk.json)](https://github.com/john-rocky/coreai-kit/actions/workflows/next-sdk-gate.yml)
[![Release](https://img.shields.io/github/v/tag/john-rocky/coreai-kit?label=release)](https://github.com/john-rocky/coreai-kit/releases)

Build LLM and computer-vision apps on Apple's Core AI framework (macOS / iOS 27 beta) in a few lines of Swift.

> Community package — not affiliated with Apple. Requires macOS 27 beta / iOS 27 beta
> (real device; the CoreAI framework is not in the iOS Simulator SDK).

```swift
import CoreAIKit

let chat = try await ChatSession(model: .qwen3_0_6B)
for try await event in chat.streamResponse(to: "Hello!") {
    if case .response(let delta) = event { print(delta, terminator: "") }
}
```

Models download automatically from the Hugging Face Hub on first use — no Python required.

## Works with Apple's FoundationModels API

`KitLanguageModel` plugs any catalog chat model into the system `LanguageModelSession` —
the same FoundationModels API you use for Apple's built-in model — and adds what the stock
`CoreAILanguageModel` adapter lacks: **tool calling** (ChatML/Hermes models) and
**guided generation** (sequential engines).

```swift
import FoundationModels
import CoreAIKit

let model = try await KitLanguageModel(model: .qwen3_0_6B)   // downloads once, then cached
let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let answer = try await session.respond(to: "What's the weather in Tokyo?")
```

Your `Tool` implementations, `@Generable` types, streaming snapshots, and transcripts work
unchanged. See `Examples/FMToolDemo` and `Examples/GuidedDemo`.

## What's inside

| Product | What it gives you |
|---|---|
| `CoreAIKit` | `ModelStore` (download/cache), `ModelCatalog` (live model list), `ChatSession` (streaming chat + live stats + guided generation), `KitLanguageModel` (FoundationModels provider with tool calling + guided generation) |
| `CoreAIKitVision` | `GraphModel` (run any `.aimodel`), `ImageTextEncoder` (CLIP), `DepthEstimator`, `CameraFeed`, image preprocessing |
| `CoreAIKitEmbeddings` | `TextEmbedder` (EmbeddingGemma, 768-d normalized) for on-device search and RAG |
| `CoreAIKitUI` | SwiftUI components: `ModelPickerBar`, `ChatTranscriptView`, `StatsBar` |

## Examples

- `Examples/ChatDemo` — multiplatform chat app (~150 lines)
- `Examples/PhotoSearch` — semantic photo search with CLIP (iOS)
- `Examples/DepthCamera` — live camera depth with Depth Anything 3 (iOS)
- `Examples/DetectCamera` — real-time object detection with RF-DETR, no NMS (iOS; nano 33–39 FPS end-to-end on iPhone 17 Pro via the zero-copy capture pipeline)
- `Examples/FMToolDemo` — local tool calling behind `LanguageModelSession` (`swift run`)
- `Examples/GuidedDemo` — guided generation: schema-valid JSON by construction (`swift run`)
- `Examples/DocChat` — on-device RAG over your notes: embeddings + retrieval tool + local LLM (`swift run`)
- `Examples/SpotlightChat` — local RAG with Apple's `SpotlightSearchTool` (WWDC26) behind your own model (`swift run`)
- `Examples/SpotlightApp` — the "ask your notes" RAG chat as a real SwiftUI app (iPhone + Mac), behind your own model
- `Examples/VisualIntel` — your own CLIP / RF-DETR behind the system **Visual Intelligence** search (iOS camera / iPad+Mac screenshot)
- `Examples/SiriAsk` — ask your local model from **Siri** (App Intents + onscreen awareness + risk-based confirmation; ≥4B)
- `Examples/VLChat` — local **VLM** image chat (Qwen3-VL) via the `KitVisionModel` vision executor (iPhone + Mac)

See `docs/GETTING_STARTED.md`.

## Requirements

- macOS 27 beta / iOS 27 beta, Xcode 27 beta
- Models run fully on device

## Versioning & stability

Tagged releases, SemVer, a pinned model catalog (each entry carries the verified
Hugging Face revision), and CI + a nightly end-to-end gate on macOS 27 beta. See
[`docs/STABILITY.md`](docs/STABILITY.md) and [`CHANGELOG.md`](CHANGELOG.md).

## License

BSD-3-Clause. See `LICENSE` and `NOTICE.txt` (portions adapted from
[apple/coreai-models](https://github.com/apple/coreai-models) and
[john-rocky/coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)).
