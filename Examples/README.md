# Examples

Every example app in one place, on two shelves:

- **Model runners** — one app per capability, driven by a **catalog id** (the id on the
  model's card). Try a model here first; each runner keeps its take-home core in
  `Sources/QuickStart.swift` — one typed function, no UI imports, the exact code you copy
  into your own app. Runners are upgraded to the id-picker + QuickStart convention
  progressively (Transcribe is the reference).
- **SDK-feature demos** — OS-integration showcases (App Intents, Spotlight, Visual
  Intelligence, FoundationModels tool calling, RAG): what you can *build around* a local
  model, not how to run one.

## Model runners

| App | Kind | What it runs |
|---|---|---|
| [Transcribe](Transcribe/) | speech-to-text | Any `asr` catalog model (Whisper / Qwen3-ASR / Parakeet): record, choose, or demo clip → transcript. GUI + `swift run transcribe-cli`. |
| [ChatDemo](ChatDemo/) | chat | Minimal multiplatform chat over `ChatSession`; pick a starter model, it downloads and streams. |
| [VLChat](VLChat/) | vision-language | Any `vlm` catalog model (Qwen3-VL 2B/4B/8B): pick a photo, ask about it. GUI + `swift run vlchat-cli`. |
| [Speak](Speak/) | text-to-speech | Any `tts` catalog model (VoxCPM): type a sentence, hear it. GUI + `swift run speak-cli`. |
| [AudioChat](AudioChat/) | audio understanding | Qwen2.5-Omni Thinker: load a clip, ask "what do you hear?" (macOS). |
| [DepthCamera](DepthCamera/) | depth | Any `depth` catalog model: live camera depth via `CameraFeed`. GUI + `swift run depth-cli`. |
| [DetectCamera](DetectCamera/) | detection | Real-time object detection on the zero-copy camera path (RF-DETR / YOLOX). GUI + `swift run detect-cli`. |
| [UpscaleDemo](UpscaleDemo/) | super-resolution | AdcSR ×4 super-resolution: pick a photo, tap Upscale. GUI + `swift run upscale-cli`. |
| [PhotoSearch](PhotoSearch/) | image-text embeddings | Semantic photo search — the photo library indexed with CLIP. |
| [DocSearch](DocSearch/) | visual doc retrieval | On-device visual document retrieval (ColModernVBERT). |

## SDK-feature demos

| App | OS surface | What it shows |
|---|---|---|
| [SiriAsk](SiriAsk/) | Siri / App Intents | "Hey Siri, Ask Gemma" → an on-device Gemma 4 answers, warm-resident. |
| [SpotlightApp](SpotlightApp/) | Spotlight indexing | Ask your own files with your own local model. |
| [SpotlightChat](SpotlightChat/) | `SpotlightSearchTool` | Local RAG through Apple's Spotlight tool (WWDC26), driven by your model. |
| [FMToolDemo](FMToolDemo/) | FoundationModels | Local tool calling behind Apple's `LanguageModelSession`. |
| [GuidedDemo](GuidedDemo/) | FoundationModels | Guided generation: logits masked through a JSON-schema grammar. |
| [DocChat](DocChat/) | — | On-device RAG end to end: notes indexed with EmbeddingGemma, answered by a local chat model. |
| [AskVLM](AskVLM/) | Visual Intelligence | Qwen3-VL as its own Visual Intelligence tab. |
| [VisualIntel](VisualIntel/) | Visual Intelligence | Your own models behind system Visual Intelligence. |

All examples build against the kit via `path: ../..` — clone the repo and they can't
version-skew. GUI apps commit their generated `.xcodeproj` (source of truth = `project.yml`):
clone → `open` → Run, no xcodegen install needed. CLI shells run headless on macOS with
`swift run` — the agent-verifiable door.
