# Cookbook — "I want to …"

Reverse lookup: find your task, copy the snippet. Everything runs fully on device
(macOS 27 beta / iOS 27 beta) and downloads its model from the Hugging Face Hub on
first use.

One product, one import: `import CoreAIOps` re-exports the model layer, so **every
snippet on this page compiles with that single import**. Most entries are **task ops**:
one line, the kit resolves and caches a catalog model behind the call, and
`options: .model("catalog-id")` swaps it. Entries that need state — a conversation, a
camera loop, a persistent index — use the **model-level APIs**; starting with an op and
dropping down later is a refactor, not a rewrite.

The "In → out" one-liners below are also API: every op carries the same summary as
`CoreAI.Op.summary` (with `defaultModelID`), so a picker or gallery UI renders straight
from `Op.allCases`.

| I want to… | Call | In → out |
|---|---|---|
| [summarize text](#work-with-text) | `CoreAI.summarize(text)` | Text → short summary |
| [pull typed values out of text](#work-with-text) | `CoreAI.extract(text, as: T.self)` | Text → typed value (`@Generable`) |
| [translate](#work-with-text) | `CoreAI.translate(text, to: .english)` | Text → translation in a named language |
| [fix grammar and typos](#work-with-text) | `CoreAI.proofread(text)` | Text → corrected text |
| [clean up a dictation transcript](#work-with-text) | `CoreAI.tidyTranscript(raw)` | Raw ASR transcript → written text |
| [redact PII](#work-with-text) | `CoreAI.redact(text)` | Text → text with PII replaced by labels |
| [find names/emails/anything in text](#work-with-text) | `CoreAI.extractEntities(from:labels:)` | Text → entities by zero-shot label |
| [chat with a local LLM, streaming](#chat-tools-and-guided-json) | `ChatSession` | Prompt ⇄ streamed conversation |
| [let the model call my functions](#chat-tools-and-guided-json) | `KitLanguageModel` + FM tools | Prompt → answer via your tools |
| [get schema-valid JSON, guaranteed](#chat-tools-and-guided-json) | guided generation | Prompt → schema-valid JSON |
| [describe a photo](#understand-images) | `CoreAI.caption(photo)` | Image → description |
| [ask questions about an image](#understand-images) | `KitVisionModel` session | Image + questions ⇄ conversation |
| [find objects in an image](#understand-images) | `CoreAI.detect(in: photo)` | Image → labeled bounding boxes |
| [get a depth map](#understand-images) | `CoreAI.estimateDepth(in: photo)` | Image → relative depth map |
| [upscale a photo ×4](#understand-images) | `CoreAI.upscale(photo)` | Image → ×4 upscaled image |
| [turn a scan into Markdown](#read-documents) | `CoreAI.read(documentAt: url)` | Document image → markdown text |
| [search my photos by meaning](#search-and-rank) | CLIP `ImageTextEncoder` | Image / text → shared vector |
| [search document pages without OCR](#search-and-rank) | `VisualDocumentRetriever` | Query + page images → ranking |
| [rank strings by meaning](#search-and-rank) | `CoreAI.search(query, in: docs)` | Query + strings → ranked matches |
| [build on-device RAG](#search-and-rank) | `TextEmbedder` + retrieval tool | Notes → grounded answers |
| [transcribe speech](#hear-and-speak) | `CoreAI.transcribe(url)` | Audio file → plain-text transcript |
| [know who said what](#hear-and-speak) | `CoreAI.transcribeMeeting(url)` | Audio file → speaker-attributed transcript |
| [describe sounds, not just words](#hear-and-speak) | `CoreAI.describeAudio(url)` | Audio file → description of the sounds |
| [speak text aloud](#hear-and-speak) | `CoreAI.speak(text)` | Text → synthesized speech (PCM) |
| [generate music from a prompt](#hear-and-speak) | `CoreAI.compose(prompt)` | Prompt → generated music (PCM) |
| [split vocals from a song](#hear-and-speak) | `CoreAI.separate(songURL)` | Song → vocal and instrumental stems |
| [recognize an action in a video](#video-and-live-camera) | `CoreAI.recognizeAction(videoAt:)` | Video clip → ranked action labels |
| [run a model on live camera frames](#video-and-live-camera) | `CameraFeed` | Camera → frame stream |
| [forecast a time series](#forecast-numbers) | `CoreAI.forecast(series)` | Number series → 128-step forecast |
| [ask my local model from Siri](#plug-into-the-system) | `Examples/SiriAsk` | Siri phrase → local model answer |
| [put my model behind Visual Intelligence](#plug-into-the-system) | `Examples/VisualIntel` / `AskVLM` | System search → your model |
| [run any `.aimodel` I exported](#escape-hatches) | `GraphModel` | Tensors in → tensors out |
| [preload models / show download progress](#first-run-downloads) | `CoreAI.prepare` / `.onDownload` | Models → downloaded ahead of use |

## First run: downloads

Every op downloads its model on first use (cached afterwards). One process-wide hook
observes all of them, and `prepare` front-loads the download + engine load behind your
loading UI so the first real call starts instantly:

```swift
CoreAI.onDownload { print("\($0.currentFile): \(Int($0.fraction * 100))%") }
try await CoreAI.prepare(.transcribe, .caption)     // any of the ops
```

## Work with text

```swift
import CoreAIOps

let tldr = try await CoreAI.summarize(article, style: .bullets)   // .concise / .oneLine
let en   = try await CoreAI.translate(review, to: .english)       // any Language("…")
let neat = try await CoreAI.proofread(draft)
let said = try await CoreAI.tidyTranscript(rawDictation)          // fillers out, numbers written
let safe = try await CoreAI.redact(supportEmail)                  // "[PERSON]", "[EMAIL]", …
let ids  = try await CoreAI.extractEntities(from: email, labels: ["person", "order number"])
```

Typed extraction uses the same `@Generable` types as Apple's FoundationModels API:

```swift
@Generable struct Order {
    @Guide(description: "Name of the ordered product") var product: String
    @Guide(description: "Number of units ordered") var quantity: Int
}
let order = try await CoreAI.extract(email, as: Order.self)
```

The free-text ops default to qwen3-4B — the floor at which translate holds up. When
speed matters more than fidelity: `options: .model("qwen3-0.6b")`. `redact` /
`extractEntities` are zero-shot (GLiNER2): pass any labels, not just the PII defaults.

`tidyTranscript` is **not** `proofread` with a different prompt — it is S1-mini by
Superwhisper, a 0.6B model trained for exactly one job, and it *deletes* what a
proofreader is contracted to keep:

```swift
try await CoreAI.tidyTranscript(
    "so um i need to like send the the report by uh friday no wait make that thursday")
// "I need to send the report by Thursday."
```

Steer it with the model's own three axes — `styling:` (`.casual` / `.semiCasual` /
`.semiFormal` / `.formal`), `structure:` (`.prose` / `.lists`), `context:` (`.general` /
`.email`) — there is no free-text instruction. English only. Filler-only input returns
the **empty string**, which is the model working, so do not retry it. Long input is cut
at word boundaries into ~450-token chunks and the rewrites are stitched: on iPhone the
engine caps prompt + generated at 1024 tokens, and a whole meeting transcript passed in
one call would stop mid-sentence.

<p align="center"><img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/pii-gliner2.jpg" alt="PII redaction on device" width="300"><br><code>CoreAI.redact</code> — GLiNER2 on iPhone</p>

## Chat, tools, and guided JSON

Streaming chat with history, live stats, and a stop button — `ChatSession`:

```swift
import CoreAIOps

let chat = try await ChatSession(model: .qwen3_0_6B)
for try await event in await chat.streamResponse(to: "What is the capital of Japan?") {
    if case .response(let delta) = event { print(delta, terminator: "") }
}
```

<p align="center"><img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/chat-youtu.gif" alt="On-device chat" width="300"><br><code>ChatSession</code> — Youtu-LLM-2B on iPhone</p>

Tool calling rides Apple's FoundationModels API — your `Tool` implementations work
unchanged behind a local model:

```swift
let model = try await KitLanguageModel(model: .qwen3_0_6B)
let session = LanguageModelSession(model: model, tools: [WeatherTool()])
let answer = try await session.respond(to: "What's the weather in Sapporo?")
```

Schema-valid JSON **by construction** (constrained decoding; needs
`engineVariant: .sequential`, ~0.6B-class models):

```swift
let model = try await KitLanguageModel(model: .qwen3_0_6B, engineVariant: .sequential)
let session = LanguageModelSession(model: model)
let plan = try await session.respond(to: "Plan a day in Kyoto.", generating: TravelPlan.self)
```

See `docs/GETTING_STARTED.md` for the support matrix and `Examples/FMToolDemo` /
`Examples/GuidedDemo` for runnable versions. `Examples/ChatDemo` is the full chat app
(~150 lines, built from `CoreAIKitUI` components).

## Understand images

```swift
import CoreAIOps

let text  = try await CoreAI.caption(imageAt: photoURL)              // or .detailed
let boxes = try await CoreAI.detect(in: photo)                       // [Detection], no NMS
let depth = try await CoreAI.estimateDepth(in: photo).cgImage()      // grayscale render
let big   = try await CoreAI.upscale(photo)                          // ×4, deterministic
```

| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/detect-rfdetr.jpg" alt="Object detection" width="240"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/depth-da3.jpg" alt="Depth estimation" width="240"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/sr-adcsr.jpg" alt="Super-resolution" width="240"> |
|:---:|:---:|:---:|
| `CoreAI.detect` — RF-DETR | `CoreAI.estimateDepth` — DA3 | `CoreAI.upscale` — AdcSR ×4 |

A conversation *about* an image (follow-up questions, one session) is the model-level
VLM behind the same FoundationModels API — attach the image to the prompt:

```swift
let vlm = try await KitVisionModel(catalog: "qwen3-vl-2b")
let session = LanguageModelSession(model: vlm)
let answer = try await session.respond(to: Prompt {
    "How many people are in this photo, and what are they doing?"
    Attachment(cgImage)
})
```

`Examples/VLChat` is the full image-chat app; `Examples/DetectCamera` runs detection at
camera rate (hold a `KitDetector` there — the op is a one-shot convenience).

## Read documents

```swift
let markdown = try await CoreAI.read(documentAt: scanURL)   // GLM-OCR, ~4 s/page
```

Tables survive as markup. Swap the engine per call: `options: .model("mineru2.5-pro")`
(2-stage layout → Markdown) or `"unlimited-ocr"`. `Examples/ReadDoc` compares them on
your own scans.

<p align="center"><img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/ocr-glm.jpg" alt="Document OCR" width="300"><br><code>CoreAI.read</code> — GLM-OCR, ~4 s/page on iPhone</p>

## Search and rank

Rank a handful of strings right now — one op:

```swift
let hits = try await CoreAI.search("refund policy", in: paragraphs, topK: 3)
```

A corpus you query repeatedly wants a persistent index — hold a `TextEmbedder`
(EmbeddingGemma, 768-d normalized) and store document vectors once:

```swift
import CoreAIOps

let embedder = try await TextEmbedder()
var docVecs: [[Float]] = []
for doc in docs { docVecs.append(try await embedder.embed(document: doc)) }  // store these
let q = try await embedder.embed(query: "refund policy")
let scores = docVecs.map { TextEmbedder.cosineSimilarity(q, $0) }  // rank by score
```

Photos by meaning (CLIP, one dot product per photo):

```swift
import CoreAIOps

let encoder = try await ImageTextEncoder()                      // CLIP ViT-B/32
let photoVec = try await encoder.encode(image: cgImage)         // store per photo
let query = try await encoder.encode(text: "red bike at the beach")
let score = ImageTextEncoder.cosineSimilarity(photoVec, query)
```

Document *pages* by meaning, no OCR (ColModernVBERT late interaction): see
`Examples/DocSearch` and `VisualDocumentRetriever`. Full RAG — retrieval as a tool the
model calls when it decides to — is `Examples/DocChat` (own index) and
`Examples/SpotlightChat` (Apple's Spotlight as the index).

## Hear and speak

```swift
import CoreAIOps

let text    = try await CoreAI.transcribe(memoURL)              // Whisper v3 turbo
let meeting = try await CoreAI.transcribeMeeting(callURL)       // diarize + per-turn ASR
print(meeting.text)                                             // one line per speaker turn
let scene   = try await CoreAI.describeAudio(clipURL)           // sounds, music, setting
let voice   = try await CoreAI.speak("The build is green.")     // PCM + sample rate
let loop    = try await CoreAI.compose("warm lo-fi loop, vinyl crackle, 90 BPM")
let stems   = try await CoreAI.separate(songURL)                // vocals / instrumental
```

`transcribe` and `transcribeMeeting` take `options: .model("parakeet-tdt-0.6b-v3")` for
faster English-family ASR, or `"qwen3-asr-1.7b"` / `"nemotron-3.5-asr-streaming-0.6b"`.
Playback / WAV writing from PCM: see `Examples/OpsDemo` (25-line WAV writer) and
`Examples/Speak` (streaming synthesis into `AVAudioEngine`).

| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/transcribe-whisper.gif" alt="Speech to text" width="280"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/diarize-sortformer.gif" alt="Speaker diarization" width="280"> |
|:---:|:---:|
| `CoreAI.transcribe` — Whisper v3 turbo | `CoreAI.transcribeMeeting` — Sortformer + ASR |

## Video and live camera

```swift
let actions = try await CoreAI.recognizeAction(videoAt: clipURL)   // V-JEPA 2
print(actions.first?.label ?? "")   // "pouring something into something"
```

Live pipelines are a for-await loop over `CameraFeed` — hold the model-level driver so
nothing reloads per frame:

```swift
import CoreAIOps

let detector = try await KitDetector(catalog: "rf-detr")
for await frame in try await CameraFeed(framesPerSecond: 30).start() {
    let boxes = try await detector.detect(in: frame)
}
```

`Examples/DetectCamera` (33–39 FPS end-to-end), `DepthCamera`, and `ActionCamera` are
the full apps.

## Forecast numbers

```swift
let f = try await CoreAI.forecast(dailySales)     // TimesFM 2.5 — ~25 ms warm
let nextWeek = f.mean.prefix(7)                   // f.quantiles for uncertainty bands
```

128 steps ahead from any univariate series; `Examples/Forecast` charts it.

<p align="center"><img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/forecast-timesfm.jpg" alt="Time-series forecasting" width="560"><br><code>CoreAI.forecast</code> — TimesFM 2.5 on iPhone</p>

## Plug into the system

- **Siri** — "ask my local model from Siri", with onscreen awareness and risk-based
  confirmation: `Examples/SiriAsk` (App Intents, ≥4B model).
- **Visual Intelligence** — your own VLM as an "ask" tab (`Examples/AskVLM`), or your
  CLIP / RF-DETR behind the system's visual search (`Examples/VisualIntel`).
- **Spotlight RAG** — Apple's `SpotlightSearchTool` behind your own model:
  `Examples/SpotlightChat` / `Examples/SpotlightApp`.

## Escape hatches

Any stateless `.aimodel` you exported runs through the generic graph runner:

```swift
import CoreAIOps

let model = try await GraphModel(contentsOf: aimodelURL, computeUnits: .neuralEngine)
let out = try await model.run(["pixel_values": .float32(pixels, shape: [1, 3, 224, 224])])
```

- A chat bundle on disk: `ChatSession(bundleAt: url)`.
- Any Hugging Face repo with the right layout: `ModelID("org/name", path: "macos")`.
- Manage the cache: `ModelStore.default.downloadedModels()` / `.delete(_:)`.
- The live model list (ids for every `options: .model(...)`):
  `await ModelCatalog.load()`, then `.available(.chat)` — or any other kind.
