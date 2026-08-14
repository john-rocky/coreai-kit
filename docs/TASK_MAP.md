# Task map — speech, image, text

What an app can do on an Apple device, what Apple already answers, what this kit answers, and
where the answer is missing. Written to decide what to build next, not to describe what exists.

**Demand column** is Swift files on GitHub calling Apple's equivalent API (measured 2026-07-28).
It is a proxy — GitHub Swift skews toward samples and tutorials, not App Store reality — but it
is revealed behaviour rather than opinion, and it is the best signal available. Sizes are the
first-use download; **(Mac)** means no iOS variant is published yet.

---

## Speech — the largest demand by a factor of two

| Task | Demand | Apple ships | Kit ships | Verdict |
|---|---|---|---|---|
| Speech → text | **5,972** | `SpeechAnalyzer` + `SpeechTranscriber` — 45 locales, OS-managed assets shared between apps, streaming, **and word timestamps** | Whisper 3.2 GB · Nemotron-streaming 1.3 GB · Parakeet 1.3 GB **(Mac)** · Qwen3-ASR 3.1 GB **(Mac)** | **Corrected 2026-08-05, from the SDK.** Two of the three advantages claimed here were already Apple's: `timeIndexedProgressiveTranscription` is a preset. What is left is a locale Apple lacks on the device in hand, behaviour that must not move under an OS update, and an offline guarantee. `CoreAI.transcribe` now defaults to Apple's and the catalog model is opt-in |
| Text → speech | **4,080** | `AVSpeechSynthesizer`, and `PersonalVoice` only for accessibility | VibeVoice 1.4 GB · VoxCPM 1.7 GB · Kokoro 341 MB **(Mac)** · VoxCPM2 5.7 GB | Apple's voices are free but cannot be cloned. **Deliberately not moved to the system backend**: doing so would leave a wrapper around a public API with no reason to exist. The clone is what earns the download |
| Who spoke when | — (no Apple API to measure) | **nothing** — `Speech.framework` contains no speaker API; the analyzer runs exactly three modules and none of them is one | Sortformer **238 MB** (measured; the 451 MB here was the macOS figure doubled) | **Apple has no answer at all.** Still true after reading the iOS 27 SDK, and now the only speech row where that holds |
| Speaker-attributed transcript | — | **nothing** | `MeetingTranscriber` (Sortformer + Apple's transcriber) = **238 MB** | Composition, not a model, and now the cheapest thing here rather than the most expensive. The thing to lead with |
| Audio understanding | — | `SNAudioStreamAnalyzer` (166, classification only) | Qwen2.5-Omni 5.5 GB **(Mac)** | Different question ("describe this audio"), no Apple equivalent, no demand signal yet |
| Source separation | 166 (nearest) | nothing | Mel-Band 470 MB | Audio, not speech. Portfolio |
| Music generation | — | nothing | Stable Audio 1.1 GB **(Mac)** | Portfolio |

**The blocking fact**: the cheap speech models are macOS-only. On iPhone the full stack is
Nemotron 2.5 GB + VibeVoice 1.4 GB + Sortformer 451 MB ≈ **4.3 GB**. With Parakeet and Kokoro
ported it would be ≈ **2.1 GB**. Nothing else on this map changes adoption economics that much
for that little work — both models are already ported, just not for iOS.

## Image and video

| Task | Demand | Apple ships | Kit ships | Verdict |
|---|---|---|---|---|
| Run a custom model | **2,388** (`VNCoreMLRequest`) | Vision + Core ML | `GraphModel` (any `.aimodel`) | Signals a real population that brings its own weights. Kit's story here is thin and undocumented |
| OCR / document → text | **3,016** | `VNRecognizeTextRequest` | GLM-OCR 1.6 GB · MinerU 2.0 GB · Unlimited-OCR 4.5 GB **(Mac)** | Apple reads text; the kit reads **structure** (tables survive as markup). That is the whole difference and it is worth stating precisely |
| Object detection | — (via Vision) | Vision (faces, rectangles, barcodes) | RF-DETR 103 MB · YOLOX-S **36 MB** | **36 MB is an ordinary app asset.** Open-vocabulary detection is not something Vision does |
| Face detection | **1,118** | `VNDetectFaceRectanglesRequest` | nothing | Apple is free, on-device, better. **Do not build this** |
| Person segmentation | 257 | `VNGeneratePersonSegmentation` | nothing | Same |
| Depth from one image | — | only `AVDepthData` from dual cameras | Depth Anything 3 **54 MB** | **Apple has no monocular depth.** 54 MB. Genuinely missing and genuinely cheap |
| Image → caption | — | nothing | **LFM2.5-VL 658 MB** · MiniCPM-V 2.1 GB · Qwen3-VL 3.3 GB | No Apple answer. 658 MB is an app-sized VLM; the bigger two buy detail |
| Screen / UI understanding | — | nothing | Holo2 5.5 GB | No Apple answer. The capability behind on-device automation |
| Image similarity search | 346 | `VNGenerateImageFeaturePrint` | CLIP 291 MB | Apple's is free and adequate for most cases |
| Super-resolution | — | `MetalFX` (games), Photos (private) | AdcSR 1.7 GB | Portfolio |
| Video → action | — | nothing | V-JEPA 2 1.4 GB | Portfolio |

## Text and language

| Task | Demand | Apple ships | Kit ships | Verdict |
|---|---|---|---|---|
| Chat / generate / summarize | **3,152** (`LanguageModelSession`, ~1 year) | Foundation Models (system model, free, 0 bytes) | **23 chat models**, 454 MB → 35 GB | Apple is free and improving. The kit wins on model choice, size class, and languages — not on convenience |
| Structured extraction | (part of the above) | `@Generable` on the system model | `CoreAI.extract` over any catalog model | Same shape, different backend |
| Translation | 924 | `TranslationSession` (iOS 17.4+) | via chat models | **Apple's is free, on-device and downloadable per language.** Hard to beat |
| Entities / NER / PII | **1,336** (`NLTagger`) | `NaturalLanguage` | GLiNER2-PII | Kit wins on **zero-shot labels** — the label set is a call-time argument, not a fixed taxonomy |
| Text embedding / semantic search | — | `NLEmbedding` (word/sentence) | EmbeddingGemma 1.2 GB · ColModernVBERT 741 MB | Apple's is weaker but free; the kit's is a real upgrade for retrieval |
| Forecasting | — | nothing | TimesFM 883 MB | No Apple answer, no measurable demand. Portfolio |

---

## Read the SDK, not this table

This file was written from measured GitHub demand and from what was believed about Apple's
stack. On 2026-08-05 the iOS 27 `Speech.framework` interface was read directly for the first
time, and two rows above were wrong in the kit's favour — streaming and word timestamps were
listed as reasons a catalog ASR model beats Apple's, and both are presets on
`SpeechTranscriber`. `SpeechDetector` also exists, which is an endpointer, i.e. the component
`SPEECH_API.md` calls "the only genuinely new component" behind `listen()`.

Nothing here is a substitute for opening the SDK before building the thing. The porting
contract's GAP gate assumes someone has.

## What the map says

**1. Half the catalog is where Apple competes hardest.** 23 of 53 entries are chat models, and
chat is the one thing Apple gives away for free with zero download. Supply is concentrated where
the differentiator is weakest.

**2. The unique positions are cheap and few.** Where Apple ships nothing at all:
diarization (**238 MB**), monocular depth (54 MB), open-vocabulary detection (36 MB), screen
understanding, document *structure*, zero-shot entity labels, and voice cloning. Five of those
are under 500 MB.

**2a. A capability Apple ships is not automatically a reason to stop — a *wrapper* is.** The
GAP gate reads as binary and is not. Apple's transcriber is as good, so the kit routes to it
and keeps the 238 MB that Apple cannot do. Apple's voices are free and cannot be cloned, so
the TTS models stay: defaulting them away would leave a wrapper around a public API. The test
is whether anything survives the routing, not whether Apple has the capability.

**3. Two tasks should never be built**: face detection and person segmentation. Apple's are
free, on-device, and better. Their demand numbers are real but they are not the kit's demand.

**4. The cheapest high-value work is an iOS port, not a new model.** Parakeet and Kokoro exist
and are macOS-only; publishing iOS variants halves the cost of the highest-demand domain.

---

## Tasks that do not exist yet, and probably should

Ranked by whether there is a mechanism pushing them, not by how interesting they are.

- **Real-time voice conversation.** `SpeechAnalyzer` at 1,608 files in its first year shows
  Apple investing in streaming speech. Streaming ASR + streaming TTS + a small chat model is a
  voice interface with no server. The kit has all three parts and no streaming API, which is
  exactly the gap in [`SHIPPABILITY_PLAN.md`](SHIPPABILITY_PLAN.md) §3.
- **Doing a chore, not answering a question.** The measured anchored-ops results show the
  ranking decouples from intelligence: a model that reasons well may fail to emit a parseable
  call. As `LanguageModelSession` adoption grows, "can it complete the task" becomes the
  question, and nobody publishes that.
- **Answering from the user's own documents.** Embedding + retrieval models are in the catalog
  but there is no op for the whole job. Apple's Spotlight and App Intents own the surrounding
  surface; the retrieval quality is the part Apple does not answer.
- **Voice that belongs to the user.** Apple's Personal Voice exists for accessibility, narrowly.
  Zero-shot cloning (VoxCPM, VibeVoice, Chatterbox in progress) is a different product, and the
  privacy argument only works on-device. **After the 2026-08-05 SDK read this is the only
  remaining reason the TTS models exist**, which makes it the row to build on rather than one
  of several.

## What this implies for structure

Speech earns its own product and its own front door: it is the largest demand, the kit's deepest
inventory, and it contains the one capability Apple has no answer for. It does **not** earn its
own repository — one maintainer, and today two pieces of split state were found to have drifted
apart silently. One repo, one catalog, one CI; a `CoreAISpeech` SwiftPM product so an adopter
pulls only what they need.

The line for that product is **the human voice**: recognition, synthesis, diarization, and the
pipelines that combine them. Music generation and source separation are audio but not speech,
have no demand signal, and stay in the general kit as portfolio.
