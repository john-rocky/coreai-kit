# CoreAIKit

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
- `Examples/FMToolDemo` — local tool calling behind `LanguageModelSession` (`swift run`)
- `Examples/GuidedDemo` — guided generation: schema-valid JSON by construction (`swift run`)
- `Examples/DocChat` — on-device RAG over your notes: embeddings + retrieval tool + local LLM (`swift run`)

See `docs/GETTING_STARTED.md`.

## Requirements

- macOS 27 beta / iOS 27 beta, Xcode 27 beta
- Models run fully on device

## License

BSD-3-Clause. See `LICENSE` and `NOTICE.txt` (portions adapted from
[apple/coreai-models](https://github.com/apple/coreai-models) and
[john-rocky/coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)).
