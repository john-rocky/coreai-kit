# Design — an op should pick the cheapest backend that does the job

Today every op resolves to a catalog model. `CoreAI.read` is GLM-OCR at 1.6 GB even when the
page is plain prose and `VNRecognizeTextRequest` — free, already installed, on every device —
would have answered it. The op layer ignores what Apple already ships and always takes the
heaviest path.

Changing that is the single largest lever on adoption, because it moves the entry cost of most
ops from gigabytes to **zero**, and the entry cost is what decides whether an engineer adds
something on a Tuesday afternoon or takes it to a meeting.

---

## What changes

Nothing in the call.

```swift
let text = try await CoreAI.read(image)
```

Underneath, the op tries the cheapest backend that satisfies the request and escalates only when
it must:

```
Apple system framework  (0 bytes, every device, immediate)
        ↓  not available, or not good enough for what was asked
catalog model           (gigabytes, better, downloads once)
```

Three consequences, and they are the point:

1. **The first call costs nothing.** An engineer can ship the feature before deciding whether to
   pay for the upgrade.
2. **Ops stop disappearing on unsupported devices.** `unsupportedDevice` becomes a fallback
   rather than an absence — an A16 phone gets Vision's answer instead of no feature.
3. **The heavy model becomes an upgrade, not a prerequisite.** The product decision moves after
   the code works instead of before it starts.

## Where Apple already has an answer

From [`TASK_MAP.md`](TASK_MAP.md), with the demand figures that make each row worth wiring.

| Op | Apple backend | Catalog backend | Wiring worth it |
|---|---|---|---|
| `transcribe` | `SFSpeechRecognizer` / `SpeechAnalyzer` | Whisper, Nemotron, Parakeet | **yes** — highest demand of all |
| `speak` | `AVSpeechSynthesizer` | Kokoro, VibeVoice, VoxCPM | **yes** — second highest |
| `summarize`, `extract`, `proofread` | Foundation Models | 23 chat models | **yes** — Apple's is free and adequate for short text |
| `read` | `VNRecognizeTextRequest` | GLM-OCR, MinerU | **yes** — but see the caveat below |
| `translate` | `TranslationSession` | chat models | **yes** — Apple's is genuinely hard to beat |
| `extractEntities`, `redact` | `NaturalLanguage` | GLiNER2 | partly — Apple's taxonomy is fixed, the model's is not |
| `search` | `NLEmbedding` | EmbeddingGemma | partly — quality gap is real |
| `detect` | Vision (faces, rectangles, barcodes) | RF-DETR, YOLOX | **no** — different job; Vision does not do open vocabulary |
| `tidyTranscript` | nothing — `NaturalLanguage` corrects spelling, it does not remove fillers or resolve false starts | S1-mini | n/a — no Apple equivalent |
| `caption`, `estimateDepth`, `upscale`, `forecast`, `compose`, `separate`, `recognizeAction` | nothing | catalog only | n/a |

## The rule that keeps this honest

**The caller must always be able to see which backend answered.**

```swift
let result = try await CoreAI.read(image)
result.text
result.backend      // .system(.vision) | .model("glm-ocr")
```

Silently returning a worse answer because it was cheaper is not a convenience, it is a defect
that surfaces as a mystery in someone's product. Any op that can route must say what it did.

And routing must be overridable in both directions:

```swift
try await CoreAI.read(image, options: .model("glm-ocr"))   // force the upgrade
try await CoreAI.read(image, options: .systemOnly)         // never download anything
```

`.systemOnly` matters more than it looks: it is how an app ships a feature with a hard
zero-bytes guarantee, which is a real product requirement and currently impossible.

## The work is measurement, not plumbing

The plumbing is a switch. **What does not exist is the basis for the decision** — for each op,
where the free backend stops being good enough.

That question has never been answered publicly for any of these pairs. It is the same kind of
work as the model measurements already done here, and it is the reason this is defensible: the
switch is easy to copy, the table behind it is not.

First measurement, because it decides the most contested row:

> **Vision OCR vs GLM-OCR, same documents, same scoring.** Vision returns words; GLM-OCR returns
> structure. The hypothesis is that the 1.6 GB buys nothing on plain prose and buys everything on
> tables and forms. If that holds, `read` routes on document type, not on quality thresholds —
> and that is a much simpler rule than a score.

Until a row has been measured, that op does not route. **A guessed threshold is worse than no
routing**, because it produces a quality difference nobody asked for and nobody can see.

## Risks worth stating before starting

- **Apple's backends have their own availability rules** — locale, device, an entitlement, a
  permission. Routing must fold those into the existing `capability(_:)` answer rather than
  discovering them at call time.
- **Two backends means two output shapes.** Vision's text is not GLM-OCR's markdown. Either the
  op normalises (losing structure the model provided) or the result type carries the difference
  honestly. The second is correct and less convenient; take it.
- **This widens the test surface.** Every routed op now has at least two paths and a decision
  between them.
- **It cuts against the catalog.** If Apple's free backend covers most calls, most users never
  download a model, and download counts fall. That is the correct outcome for adopters and worth
  naming so it does not read as a regression later.

## Order

1. Result types carry `.backend`, and `OpOptions` gains `.systemOnly` and model forcing. No
   routing yet — just the surface that makes routing describable.
2. Measure one pair (Vision vs GLM-OCR). Publish it; it is a knowledge-base note either way.
3. Route the ops whose free backend is unambiguously adequate for the common case:
   `translate`, `speak`, `summarize`.
4. Route the measured ones as the measurements land.
5. Leave `detect`, `caption`, `estimateDepth` and the rest catalog-only — Apple has no answer,
   so there is nothing to route to.
