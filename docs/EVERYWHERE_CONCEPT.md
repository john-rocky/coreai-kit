# CoreAI Everywhere — product concept

> **Uses Apple's when Apple's is better — measured, not assumed.**

One API where every on-device AI task on Apple platforms has an answer: Apple's own framework
when it wins, an open model when it does not, and something where Apple ships nothing at all.

Concept and structure only. Implementation is a separate session; the measurement it depends on
is `~/code/coreai/APPLE_STACK_BENCH_KICKOFF.md`.

---

## The problem, from the app engineer's side

Apple's on-device AI is **seven unrelated frameworks** — Vision, Speech, AVSpeechSynthesizer,
NaturalLanguage, Translation, Foundation Models, SoundAnalysis. Different API shapes, different
result types, different availability rules. "Transcribe this, then summarise it" is written
against two frameworks with nothing in common.

And there are holes: no speaker diarization, no monocular depth, no open-vocabulary detection,
no document *structure*, no caller-defined entity labels.

Today's ops layer covers some of the holes and **ignores Apple entirely** — `CoreAI.read` is
1.6 GB of GLM-OCR even when the page is prose and `VNRecognizeTextRequest`, free and already
installed, would have answered it.

## The product

Three tiers behind one surface. The third tier is what makes it *integrated* rather than
*another library*.

| Tier | Tasks | Backend | Cost |
|---|---|---|---|
| **Apple only** | face detection, barcodes, rectangles, saliency | thin wrapper over Vision | ~0 |
| **Both — routed by measurement** | OCR, speech→text, text→speech, chat/summarise, translate, entities, embeddings | Apple's or a model, decided by the benchmark | 0 until it escalates |
| **Model only** | diarization, monocular depth, open-vocabulary detection, captioning, VLM, super-resolution, forecasting, music, separation, action | catalog model | download |

**Tier 1 is the point.** If an engineer has to leave this API and learn Vision to detect a face,
it is not integrated. Wrapping it costs almost nothing and buys the sentence *everything is
here*.

Three consequences:

1. **The first call can cost zero bytes.** Ship the feature, decide later whether to pay for the
   upgrade.
2. **Features stop vanishing on older devices.** `unsupportedDevice` becomes a fallback, not an
   absence — an A16 phone gets Vision's answer instead of nothing.
3. **The heavy model becomes an upgrade, not a prerequisite.** The product decision moves after
   the code works instead of before it starts.

## Two rules that keep it honest

**The caller can always see which backend answered.**

```swift
let r = try await CoreAI.read(image)
r.text
r.backend        // .system(.vision) | .model("glm-ocr")
```

Silently returning a cheaper, worse answer is not convenience; it is a defect that surfaces as a
mystery in someone else's product.

**An op does not route until its pair has been measured.** A guessed threshold produces a
quality difference nobody asked for and nobody can see. Until `APPLE_STACK_BENCH` has covered a
row, that op keeps its current single backend.

## The dependency structure — measured, and it changes the design

The weight is not where it looks. **Apple's frameworks are system frameworks**: linking Vision,
Speech, NaturalLanguage or Translation is dynamically resolved from the OS and adds
approximately nothing to an app binary. **The whole Apple side of this product is nearly free.**

What is expensive is already here:

| Dependency | Size | Comes in via |
|---|---|---|
| swift-nio | 199,000 lines | swift-transformers → Hub → swift-huggingface |
| swift-crypto | 111,000 lines | same |
| xgrammar (C++) | 88,000 lines | guided generation |
| swift-huggingface | 26,000 lines | swift-transformers |
| swift-transformers | 15,000 lines | tokenizers, chat templates |

Roughly **440,000 lines of third-party code**, all of it on the LLM path.

**There is no cheap way to cut it.** The kit imports `Tokenizers` in 40 places and `Hub`,
`Generation` and `Models` in none — but upstream declares
`.target(name: "Tokenizers", dependencies: ["Hub", …])`, so narrowing the product dependency
from `Transformers` to `Tokenizers` drops two targets and keeps the whole heavy chain. Removing
it for real means replacing the tokenizer, which is its own project.

### But a light path already exists, and nothing advertises it

```
CoreAIKitCore        no third-party deps at all — the Hub client is URLSession, Foundation only
├── CoreAIKitVision  depends only on Core → also zero third-party
└── CoreAIKit (LLM)  Transformers → Hub → huggingface / crypto / xet → the 440k lines
     └── CoreAIOps   everything
```

An app doing detection (36–103 MB), depth (54 MB), CLIP (291 MB) or running any `.aimodel`
through `GraphModel` links **zero third-party code today**.

The problem is that the documented entry point is the heaviest one. README and `AGENTS.md` both
say `import CoreAIOps` is the quick path, and it pulls all of it. **The light path exists and is
undocumented.**

### What the product adds

```
CoreAIKitCore          zero deps
├── CoreAISystem  NEW  Apple frameworks only → zero deps
├── CoreAIKitVision    zero deps (unchanged)
├── CoreAIKit (LLM)    heavy, unchanged
├── CoreAIOpsLite NEW  system + vision tiers → zero third-party, zero download
└── CoreAIOps          all tiers (unchanged surface)
```

`CoreAISystem` is tier 1 and the Apple half of tier 2. Because it touches only system
frameworks, it carries **no third-party dependency at all** — and it is the same code the
integration needs anyway.

So the answer to "does integrating make the library heavier" is the opposite of what it looks
like: **integration is what finally makes a light path selectable.** Today there is no way to
say "AI features, nothing downloaded" — after this, `import CoreAIOpsLite` is exactly that.

An escape hatch in both directions, on the full product:

```swift
try await CoreAI.read(image, options: .systemOnly)      // never download anything
try await CoreAI.read(image, options: .model("glm-ocr")) // force the upgrade
```

## What is new versus today's ops

| | Today | After |
|---|---|---|
| Backends | catalog model, always | measured choice, disclosed |
| Apple-only tasks | absent | covered (tier 1) |
| Unsupported device | feature disappears | falls back |
| First-use cost | gigabytes | zero for most ops |
| Light path | exists, undocumented | a product you can import |
| Surface | 21 ops | same ops, more of them answerable |

The call sites do not change. `CoreAI.read(image)` is the same line before and after.

## Risks, stated before starting

- **Two backends means two output shapes.** Vision's text is not GLM-OCR's markdown. Either the
  op normalises — losing the structure the model provided — or the result type carries the
  difference honestly. Take the second; it is correct and less convenient.
- **Apple's backends have their own availability rules** (locale, device, permission,
  entitlement). These must fold into `capability(_:)` rather than being discovered at call time.
- **The test surface doubles** for every routed op.
- **It cuts against the catalog.** If Apple's free backend covers the common case, most users
  never download a model and the download counts fall. That is the right outcome for adopters
  and should be said out loud rather than discovered later as a regression.
- **Naming.** `CoreAI Everywhere` is a positioning name for the umbrella; the existing
  `CoreAIKit` / `CoreAIOps` / `coreai-model-zoo` family stays. Adding a second brand would
  dilute search results and cross-links that are currently working.

## Order

1. `CoreAISystem` — tier 1 wrappers, zero dependencies. Buildable today, needs no measurement.
2. `CoreAIOpsLite` — the light product, and documenting that it exists.
3. `result.backend` and `OpOptions.systemOnly` — the surface that makes routing describable,
   before any routing happens.
4. Route the rows as `APPLE_STACK_BENCH` measures them, starting with OCR.
5. Tier 3 is already built; it only gains the `.backend` disclosure.
