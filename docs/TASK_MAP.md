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
| Speech → text | **5,972** | `SFSpeechRecognizer`, `SpeechAnalyzer` (iOS 26) | Whisper 3.2 GB · Nemotron-streaming 2.5 GB · Parakeet 1.3 GB **(Mac)** · Qwen3-ASR 3.1 GB **(Mac)** | Apple is free and good. Kit wins on languages, timestamps, and running where Apple's is unavailable — **but only if it can be afforded** |
| Text → speech | **4,080** | `AVSpeechSynthesizer` | VibeVoice 1.4 GB · VoxCPM 1.7 GB · Kokoro 341 MB **(Mac)** · VoxCPM2 5.7 GB | Apple's voices are robotic and cannot be cloned. **Clearest quality upgrade in the whole catalog** |
| Who spoke when | — (no Apple API to measure) | **nothing** | Sortformer 451 MB | **Apple has no answer at all.** The one speech capability that is not a better version of something Apple ships |
| Speaker-attributed transcript | — | **nothing** | `MeetingTranscriber` (Sortformer + ASR) | Composition, not a model. The thing to lead with |
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
| Image → caption | — | nothing | Qwen3-VL 3.3 GB · MiniCPM-V 2.1 GB | No Apple answer; expensive |
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

## What the map says

**1. Half the catalog is where Apple competes hardest.** 23 of 53 entries are chat models, and
chat is the one thing Apple gives away for free with zero download. Supply is concentrated where
the differentiator is weakest.

**2. The unique positions are cheap and few.** Where Apple ships nothing at all:
diarization (451 MB), monocular depth (54 MB), open-vocabulary detection (36 MB), screen
understanding, document *structure*, zero-shot entity labels. Four of those are under 500 MB.

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
  privacy argument only works on-device.

## What this implies for structure

Speech earns its own product and its own front door: it is the largest demand, the kit's deepest
inventory, and it contains the one capability Apple has no answer for. It does **not** earn its
own repository — one maintainer, and today two pieces of split state were found to have drifted
apart silently. One repo, one catalog, one CI; a `CoreAISpeech` SwiftPM product so an adopter
pulls only what they need.

The line for that product is **the human voice**: recognition, synthesis, diarization, and the
pipelines that combine them. Music generation and source separation are audio but not speech,
have no demand signal, and stay in the general kit as portfolio.
