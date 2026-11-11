# Use your model with FoundationModels

`KitLanguageModel` adapts **chat bundles** to Apple's `LanguageModelSession`. The
catalog also contains other task kinds; vision uses `KitVisionModel`, and speech has
its own API. For a repeatable two-turn example, select this release's built-in pin:

```swift
import FoundationModels
import CoreAIKit

guard let modelID = ModelCatalog.builtin.entry(id: "qwen3-0.6b")?.modelID else {
    throw CoreAIKitError.modelNotAvailableOnPlatform(id: "qwen3-0.6b")
}
let model = try await KitLanguageModel(model: modelID)
let session = LanguageModelSession(model: model)
let first = try await session.respond(to: "Remember: my secret word is ORCHID. Confirm briefly.")
let second = try await session.respond(to: "What is my secret word? Reply with only that word.")
print(first.content, second.content)
```

The second answer should recall ORCHID. Tool calling requires a compatible
ChatML/Hermes model; guided generation requires a sequential engine. See the
[capability limits](#works-with-apples-foundationmodels-api).

After chat, [Speak](../Examples/Speak) adds one concrete capability: VoxCPM 0.5B text to
speech, producing a mono WAV with a fixed voice. It has a separate model download.

## Works with Apple's FoundationModels API

`KitLanguageModel` plugs compatible Core AI chat bundles into the system `LanguageModelSession` —
the same FoundationModels API you use for Apple's built-in model — and adds what the stock
`CoreAILanguageModel` adapter lacks: **tool calling** (ChatML/Hermes models and LFM3) and
**guided generation** (sequential engines).

```swift
import FoundationModels
import CoreAIKit

guard let modelID = ModelCatalog.builtin.entry(id: "qwen3-0.6b")?.modelID else {
    throw CoreAIKitError.modelNotAvailableOnPlatform(id: "qwen3-0.6b")
}
let model = try await KitLanguageModel(model: modelID)   // downloads once, then cached
let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let answer = try await session.respond(to: "What's the weather in Tokyo?")
```

`KitVisionModel` does the same for vision-language models — attach an image to the prompt:

```swift
let vlm = try await KitVisionModel(catalog: "qwen3-vl-2b")   // decoder + vision tower
let session = LanguageModelSession(model: vlm)
let answer = try await session.respond(to: Prompt {
    Attachment(cgImage)
    "What is in this photo?"
})
```

Your `Tool` implementations, `@Generable` types, streaming snapshots, and transcripts work
unchanged. See `Examples/FMToolDemo`, `Examples/GuidedDemo`, and `Examples/VLChat`.

What each provider honestly advertises:

| | `KitLanguageModel` (text) | `KitVisionModel` (VL) |
|---|---|---|
| Tool calling | ChatML/Hermes models — the qwen3 family — and LFM3's XML calls (LFM2.5's pythonic dialect is not parsed) | not in v1, by design |
| Reasoning | thinking models stream `.reasoning` | Qwen3-VL thinks by default |
| Guided generation | sequential engines only (`engineVariant: .sequential`) | not in v1, by design |
| Vision | — | one image per session; every turn re-prefills the full prompt (the vision encode is reused while the image is unchanged) |

Compared with Apple's stock `CoreAILanguageModel` adapter, this provider adds tool
calling, per-turn usage events (including `Usage.Input.cachedTokenCount`), and a KV
fast path that rewinds to the longest shared prefix with the previous turn
(`reset(to:)` + the engine's implicit prefix caching) instead of re-prefilling the
whole transcript — including across a divergence, e.g. a re-rendered transcript.
