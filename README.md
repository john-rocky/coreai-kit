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

## See it running

Real-device captures (iPhone 17 Pro / M4 Max), everything fully on-device. Each cell links to
the kit example — or zoo app — that runs the same model. (Media lives in
[coreai-assets](https://github.com/john-rocky/coreai-assets), so cloning this repo stays fast.)

| | | |
|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/chat-youtu.gif" alt="On-device chat" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/transcribe-whisper.gif" alt="Speech-to-text" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/diarize-sortformer.gif" alt="Speaker diarization" width="250"> |
| Chat — Youtu-LLM-2B<br>[`ChatDemo`](Examples/ChatDemo) | Speech→text — Whisper v3 turbo<br>[`Transcribe`](Examples/Transcribe) | Who-said-what — Sortformer + Parakeet<br>[`Meeting`](Examples/Meeting) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/vlm-holo2.gif" alt="Computer-use VLM" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/agent-fastcontext.gif" alt="Repo-exploration agent" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/detect-rfdetr.jpg" alt="Object detection" width="250"> |
| Screen VLM — Holo2-4B<br>[`VLChat`](Examples/VLChat) | Repo agent — FastContext-4B<br>[zoo `CoreAIChat`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIChat) | Detection — RF-DETR nano, no NMS<br>[`DetectCamera`](Examples/DetectCamera) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/seg-sam3.jpg" alt="Promptable segmentation" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/depth-da3.jpg" alt="Depth estimation" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/sr-adcsr.jpg" alt="Super-resolution" width="250"> |
| Segmentation — SAM 3<br>[zoo](https://github.com/john-rocky/coreai-model-zoo) | Depth — Depth Anything 3<br>[`DepthCamera`](Examples/DepthCamera) | 4× super-res — AdcSR<br>[`UpscaleDemo`](Examples/UpscaleDemo) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/pii-gliner2.jpg" alt="PII redaction" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/ocr-glm.jpg" alt="Document OCR" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/bitcpm.gif" alt="Ternary LLM" width="250"> |
| PII redaction — GLiNER2<br>[`InfoExtract`](Examples/InfoExtract) | Doc OCR — GLM-OCR 0.9B, ~4 s/page<br>[`ReadDoc`](Examples/ReadDoc) | 1.58-bit ternary — BitCPM-8B in ~2.1 GB<br>[zoo `CoreAIChat`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIChat) |

| | |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/imagegen-glm.jpg" alt="Text-to-image" width="390"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/edit-flux.jpg" alt="In-context image editing" width="390"> |
| Text→image — GLM-Image<br>[zoo `CoreAIImageGen`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIImageGen) | In-context edit — FLUX.2 klein<br>[zoo `CoreAIImageGen`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIImageGen) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/t2v-ltx.gif" alt="Text-to-video" width="390"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/splat-tripo.gif" alt="Photo to 3D gaussian splat" width="390"> |
| Text→video — LTX-Video 2B<br>[zoo `CoreAIVideo`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIVideo) | Photo→3D splat — TripoSplat<br>[zoo `TripoSplatMac`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/TripoSplatMac) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/dllm-llada.gif" alt="Diffusion LLM" width="390"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/readdoc-mineru.gif" alt="Document parsing" width="390"> |
| Diffusion LLM (parallel denoise) — LLaDA-8B<br>[`DiffuseChat`](Examples/DiffuseChat) | Doc→Markdown — MinerU2.5<br>[`ReadDoc`](Examples/ReadDoc) |

<p align="center"><img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/forecast-timesfm.jpg" alt="Time-series forecasting" width="720"><br>Time-series forecasting — TimesFM 2.5, ~25 ms/forecast on iPhone · <a href="Examples/Forecast"><code>Forecast</code></a></p>

<p align="center"><img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/coder-ornith.gif" alt="Agentic coding on Mac" width="720"><br>Agentic coding — Ornith-1.0-9B on M4 Max · <a href="https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIChatMac">zoo <code>CoreAIChatMac</code></a></p>

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

`KitVisionModel` does the same for vision-language models — attach an image to the prompt:

```swift
let vlm = try await KitVisionModel(catalog: "qwen3-vl-2b")   // decoder + vision tower
let session = LanguageModelSession(model: vlm)
let answer = try await session.respond(to: Prompt {
    "What is in this photo?"
    Attachment(cgImage)
})
```

Your `Tool` implementations, `@Generable` types, streaming snapshots, and transcripts work
unchanged. See `Examples/FMToolDemo`, `Examples/GuidedDemo`, and `Examples/VLChat`.

What each provider honestly advertises:

| | `KitLanguageModel` (text) | `KitVisionModel` (VL) |
|---|---|---|
| Tool calling | ChatML/Hermes models — the qwen3 family (LFM's pythonic dialect is not parsed) | not in v1, by design |
| Reasoning | thinking models stream `.reasoning` | Qwen3-VL thinks by default |
| Guided generation | sequential engines only (`engineVariant: .sequential`) | not in v1, by design |
| Vision | — | one image per session; every turn re-prefills the full prompt (the vision encode is reused while the image is unchanged) |

Compared with Apple's stock `CoreAILanguageModel` adapter, this provider adds tool
calling, per-turn usage events (including `Usage.Input.cachedTokenCount`), and an
append-only KV fast path that reuses the previous turn's cache instead of re-prefilling
the whole transcript.

## What's inside

| Product | What it gives you |
|---|---|
| `CoreAIKit` | `ModelStore` (download/cache), `ModelCatalog` (live model list), `ChatSession` (streaming chat + live stats + guided generation), `KitLanguageModel` (FoundationModels provider with tool calling + guided generation) |
| `CoreAIKitVision` | `GraphModel` (run any `.aimodel`), `ImageTextEncoder` (CLIP), `DepthEstimator`, `CameraFeed`, image preprocessing |
| `CoreAIKitEmbeddings` | `TextEmbedder` (EmbeddingGemma, 768-d normalized) for on-device search and RAG |
| `CoreAIKitUI` | SwiftUI components: `ModelPickerBar`, `ChatTranscriptView`, `StatsBar` |
| `CoreAIOps` | Anchored task-level ops — `CoreAI.summarize`, `CoreAI.extract` (typed by `@Generable`) — that resolve a catalog model behind a stable API |

## Examples

Text & chat

- `Examples/ChatDemo` — multiplatform chat app (~150 lines)
- `Examples/DiffuseChat` — diffusion LLM chat: watch LLaDA-8B denoise all tokens in parallel (Mac)
- `Examples/FMToolDemo` — local tool calling behind `LanguageModelSession` (`swift run`)
- `Examples/GuidedDemo` — guided generation: schema-valid JSON by construction (`swift run`)
- `Examples/OpsDemo` — task-level ops: `CoreAI.summarize` / `CoreAI.extract`, no sessions or prompts in app code (`swift run`)
- `Examples/InfoExtract` — schema-driven extraction / PII redaction with GLiNER2 (iPhone + Mac)

Vision

- `Examples/VLChat` — local **VLM** image chat (Qwen3-VL) via the `KitVisionModel` vision executor (iPhone + Mac)
- `Examples/AskVLM` — your Qwen3-VL as its own **Visual Intelligence** tab (offline "ask")
- `Examples/VisualIntel` — your own CLIP / RF-DETR behind the system **Visual Intelligence** search (iOS camera / iPad+Mac screenshot)
- `Examples/PhotoSearch` — semantic photo search with CLIP (iOS)
- `Examples/DetectCamera` — real-time object detection with RF-DETR, no NMS (iOS; nano 33–39 FPS end-to-end on iPhone 17 Pro via the zero-copy capture pipeline)
- `Examples/DepthCamera` — live camera depth with Depth Anything 3 (iOS)
- `Examples/UpscaleDemo` — one-step diffusion super-resolution with AdcSR
- `Examples/ActionCamera` — video action recognition with V-JEPA 2 (world model)
- `Examples/ReadDoc` — whole-page document OCR → Markdown (GLM-OCR / MinerU2.5)
- `Examples/DocSearch` — visual document retrieval, no OCR (ColModernVBERT late interaction)

Audio & speech

- `Examples/Transcribe` — speech→text (Whisper large-v3-turbo, Qwen3-ASR, Parakeet TDT)
- `Examples/Meeting` — who-said-what: Sortformer diarization + per-turn ASR in one API
- `Examples/Speak` — text-to-speech (Kokoro, VoxCPM)
- `Examples/Music` — text→music with Stable Audio Open Small (~12× realtime on iPhone)
- `Examples/AudioChat` — audio *understanding* — describe sounds, not just transcripts (Qwen2.5-Omni)

RAG, agents & system integration

- `Examples/DocChat` — on-device RAG over your notes: embeddings + retrieval tool + local LLM (`swift run`)
- `Examples/SpotlightChat` — local RAG with Apple's `SpotlightSearchTool` (WWDC26) behind your own model (`swift run`)
- `Examples/SpotlightApp` — the "ask your notes" RAG chat as a real SwiftUI app (iPhone + Mac), behind your own model
- `Examples/SiriAsk` — ask your local model from **Siri** (App Intents + onscreen awareness + risk-based confirmation; ≥4B)

Other modalities

- `Examples/Forecast` — time-series forecasting with TimesFM 2.5 (~25 ms/forecast on iPhone)

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
