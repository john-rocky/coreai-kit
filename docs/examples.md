# Examples

Task ops

- `Examples/OpsGallery` — every one-shot op as a card: pick an input, run the one-line call, see the result (iPhone + Mac)
- `Examples/OpsDemo` — task-level ops: one voice memo → transcript, summary, typed action items, translation, spoken reply; or one image → caption, detections, OCR (`swift run`)

Text & chat

- `Examples/ChatDemo` — multiplatform chat app (~150 lines)
- `Examples/DiffuseChat` — diffusion LLM chat: watch LLaDA-8B denoise all tokens in parallel (Mac)
- `Examples/FMToolDemo` — local tool calling behind `LanguageModelSession` (`swift run`)
- `Examples/GuidedDemo` — guided generation: schema-valid JSON by construction (`swift run`)
- `Examples/InfoExtract` — schema-driven extraction / PII redaction with GLiNER2 (iPhone + Mac)
- `Examples/TextClassify` — zero-shot text classification: labels named at call time, several questions in one forward (GLiNER2.5-Decide, `swift run`)

Typed decisions — the System One shape, on device

- `Examples/Decide` — a text and a typed question in, the answer with its probability out, nothing generated; a chat model zero-shot (`minicpm5-2b`) or a model trained for decisions (`decider-0.8b`; `openthai-systemone` — Thai + English, up to 255 options, an abstain probability; `apus-decision-v1-4b` — browser actions and workflow steps, English + Chinese, Mac; `qwen3.5-2b-decision` — a calibrated 2B, English, plain-text prompt, iPhone-sized; `system-one-scorer-4b` — a scoring head, one row per option, English, Mac, CC BY-NC 4.0; `decider-2b-vision` — questions about an image, `CoreAI.decide(image:…)`, every question read in one pass; `clef-flash` — every question of a request read at once by a joint head, a text, JSON or one image as the state, Mac, its own catalog kind `jointDecision`: `KitClefDecider`, `decide-cli` and `/v1/systemone` with `images`; `kev-0.8b` and `kev-4b` — Jared Palmer's decision models, each question its own row read by a pointer head, a text or JSON state, 0.8B on iPhone and Mac, 4B on Mac, their own catalog kind `rowDecision`: `KitKevDecider`, `decide-cli` and `/v1/systemone`; `d1-omni-600m` — Liquid AI's decision model over a text, JSON, images or a voice clip, each question its own row read at its option markers, iPhone and Mac, its own catalog kind `omniDecision`: `KitD1OmniDecider`, `decide-cli` and `/v1/systemone` with `images` or `audio`; `d1-3b` — Liquid AI's 3B decision model over a text, JSON or pictures, each question its own row read at its last token, the Mac's fp16 decoder and the iPhone's int8 one (the iPhone app needs increased-memory-limit), its own catalog kind `tokenDecision`: `KitD1Decider`, `decide-cli` and `/v1/systemone` with `images`). Ten whole uses from the same sources on iPhone and Mac: copy an email and a checkout form fills at once, a contract read as a checklist, a folder sorted by what needs you, a car the model drives lane by lane, a CSV with the columns you ask for, a command guard for a coding agent (also a Claude Code hook), tool results dropped from an agent's context by relevance, tone / intent / emoji as you type (`swift run decide-cli` is the headless door; `decide-cli serve` is a `/v1/systemone` endpoint for a client written for the hosted API)

Vision

- `Examples/VLChat` — local **VLM** image chat (Qwen3-VL) via the `KitVisionModel` vision executor (iPhone + Mac)
- `Examples/AskVLM` — your Qwen3-VL as its own **Visual Intelligence** tab (offline "ask")
- `Examples/VisualIntel` — your own CLIP / RF-DETR behind the system **Visual Intelligence** search (iOS camera / iPad+Mac screenshot)
- `Examples/PhotoSearch` — semantic photo search with CLIP (iOS)
- `Examples/DetectCamera` — real-time object detection with RF-DETR, no NMS (iOS; nano 33–39 FPS end-to-end on iPhone 17 Pro via the zero-copy capture pipeline)
- `Examples/LiveCamera` — the four live tasks as four tabs: `watch()`, `watchDepth()`, a trigger gating a VLM, and `scan(videoAt:)` over a video file, with the measured stats and thermal governor on screen (iOS; `swift run live-cli` covers the offline half with no device)
- `Examples/DepthCamera` — live camera depth with Depth Anything 3 (iOS)
- `Examples/UpscaleDemo` — one-step diffusion super-resolution with AdcSR
- `Examples/ActionCamera` — video action recognition with V-JEPA 2 (world model)
- `Examples/ReadDoc` — whole-page document OCR → Markdown (GLM-OCR / MinerU2.5)
- `Examples/DocSearch` — visual document retrieval, no OCR (ColModernVBERT late interaction)

Audio & speech

- `Examples/Transcribe` — speech→text (Whisper large-v3-turbo, Qwen3-ASR, Parakeet TDT)
- `Examples/Tidy` — the other half of dictation: raw transcript → written text (S1-mini by Superwhisper — fillers, false starts, spoken numbers and dates)
- `Examples/Meeting` — who-said-what: Sortformer (4 speakers) or Nemotron-3-Diarization (8)
  diarization + per-turn ASR in one API
- `Examples/Speak` — text-to-speech (Kokoro, VoxCPM)
- `Examples/Music` — text→music with Stable Audio Open Small (~12× realtime on iPhone)
- `Examples/AudioChat` — audio *understanding* — describe sounds, not just transcripts (Qwen2.5-Omni)

Also in the audio surface, without a dedicated example yet: `KitDialogue` (multi-speaker /
podcast-style TTS — `perform("Speaker 1: …\nSpeaker 2: …")`, VibeVoice-Realtime-0.5B) and
`KitSeparator` (song → vocals + instrumental stems, Mel-Band RoFormer).

RAG, agents & system integration

- `Examples/DocChat` — on-device RAG over your notes: embeddings + retrieval tool + local LLM (`swift run`)
- `Examples/SpotlightChat` — local RAG with Apple's `SpotlightSearchTool` (WWDC26) behind your own model (`swift run`)
- `Examples/SpotlightApp` — the "ask your notes" RAG chat as a real SwiftUI app (iPhone + Mac), behind your own model
- `Examples/SiriAsk` — ask your local model from **Siri** (App Intents + onscreen awareness + risk-based confirmation; ≥4B)

Other modalities

- `Examples/Forecast` — time-series forecasting of your own series (paste or CSV) with TimesFM 2.5 (~25 ms/forecast on iPhone) or Timestral 209M (21 quantiles, horizons up to 1,024 steps)

See `docs/GETTING_STARTED.md`.

Also in this folder

- `Examples/Cutout` — cut out what you name in a photo, with its soft edge, on the device
- `Examples/Look` — ask about a photo, or have the camera watch for what you name, on the device
- `Examples/ScanToType` — a document photo becomes a value of your own type
- `Examples/Week` — your week, planned on device

## What's inside

| Product | What it gives you |
|---|---|
| `CoreAIKit` | `VoiceActivityDetector` (where speech starts and stops), `ModelStore` (download/cache), `ModelCatalog` (live model list), `ChatSession` (streaming chat + live stats + guided generation), `KitLanguageModel` (FoundationModels provider with tool calling + guided generation) |
| `CoreAIKitVision` | `GraphModel` (run any `.aimodel`), `ImageTextEncoder` (CLIP), `DepthEstimator`, `CameraFeed`, `LiveVision` (camera → model, with the frame policy and thermal governor already written), `KitTracker` (detections → stable ids across frames), image preprocessing |
| `CoreAIKitEmbeddings` | `TextEmbedder` (EmbeddingGemma, 768-d normalized) for on-device search and RAG |
| `CoreAIKitUI` | SwiftUI components: `ModelPickerBar`, `ChatTranscriptView`, `StatsBar` |
| `CoreAIOps` | Twenty-five anchored task-level ops — text (`CoreAI.summarize`, `.extract` typed by `@Generable`, `.translate`, `.proofread`, `.tidyTranscript` (raw dictation → written text), `.redact`, `.decide` (typed yes/no, choice and score decisions with probabilities, nothing generated — about a text, or an image with `decide(image:…)`)), audio (`.transcribe`, `.transcribeMeeting`, `.describeAudio`, `.speak`, `.compose`, `.separate`), image (`.caption`, `.detect`, `.read`, `.upscale`, `.estimateDepth`), plus `.recognizeAction`, `.search`, `.forecast` — each resolving a catalog model behind a stable API ([Cookbook](COOKBOOK.md)). Live camera: `CoreAI.watch()` / `.watchDepth()` per frame, `CoreAI.watch(for: .label("person"))` to run an expensive model only on the frames that matter |

Beyond this package: [**coreai-model-zoo**](https://github.com/john-rocky/coreai-model-zoo) is
where the models and their conversion recipes live, and
[**awesome-core-ai**](https://github.com/john-rocky/awesome-core-ai) tracks the wider Core AI
ecosystem — Apple's own tooling, other people's converters, sample apps, and benchmarks.

## Projects using CoreAIKit

- [AnyDecisionModel](https://github.com/mattt/AnyDecisionModel) — a Swift package for typed
  decisions from language models (probabilities, choices, and scores). Since 0.4.0 its
  `CoreAIDecisionModel`, enabled by the package's `CoreAI` trait, runs a Core AI language
  bundle through CoreAIKit's `TypedDecisions` API
  ([release notes](https://github.com/mattt/AnyDecisionModel/releases/tag/0.4.0)).
