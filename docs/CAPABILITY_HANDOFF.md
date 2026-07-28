# Handoff — `CoreAI.capability(_:)`, so a non-ML engineer can ship

**Goal**: an iOS engineer who has never trained a model can put an on-device feature into a
shipping app without learning what a checkpoint, a quantization or a Hugging Face variant is.

The op layer already gets this right — the API is task-shaped, and no call requires naming a
model. What is missing is everything *around* the call. This document is the plan for closing
that, and the reasoning behind it, so the implementing session does not have to re-derive it.

---

## The gap, stated precisely

Before shipping a feature, an app engineer must answer four questions. Three of them have no
API today.

| Question | Today |
|---|---|
| Will this run on the user's device? | **nothing** — no `isSupported`, no architecture gate anywhere in `Sources/` |
| How large is the first-use download? | **nothing** — `sizeMB` is in `catalog.json` but no op exposes it |
| Is there room on disk for it? | **nothing** — a multi-GB download starts and fails partway |
| What do I show while it downloads? | `CoreAI.onDownload { … }` ✅ |

The error surface has the same shape. `CoreAIKitError` cases are
`notAHuggingFaceRepo`, `variantNotFound`, `modelNotInCatalog`, `catalogKindMismatch` — the
vocabulary of someone who knows how a model repository is laid out. There is **no case at all**
for the three failures a real app actually hits: unsupported device, insufficient storage,
insufficient memory.

Why this has stayed invisible: none of it fires on the developer's own machine. The device is
supported, the disk has room, and the model is already cached from the last run. It fires on
users' devices, after shipping.

`CoreAI.prepare(.transcribe)` exists, but it is an instruction ("fetch this now"), not a
question ("can this happen?"). **The kit can be told to act and cannot be asked.**

---

## What to build

```swift
public enum Capability: Sendable {
    case ready                                   // cached; the call will just run
    case needsDownload(bytes: Int64)             // ready after a download of this size
    case insufficientStorage(needs: Int64, free: Int64)
    case unsupportedDevice(reason: String)
}

public static func capability(_ op: Op, options: OpOptions = OpOptions()) async -> Capability
```

Design notes for whoever implements it:

- **Take the existing `Op` enum.** `CoreAI+Prepare.swift` already maps every op to its backing
  catalog model (`case .summarize, .extract, .translate, .proofread:` → `defaultModel`, and so
  on). `capability` walks the same switch. Do not build a second mapping — one of them will
  drift, and it will be this one.
- **`async`, not `throws`.** A capability query has no failure mode worth throwing: "I could not
  determine it" is `unsupportedDevice(reason:)` with an honest reason. An engineer branching on
  this in `.task { }` should not have to write a `catch` that has nothing to do.
- **Report bytes, not a formatted string.** The caller decides whether that is "1.6 GB" or a
  progress bar; the API should not choose a locale or a unit.
- **Cached check must not touch the network.** `ModelStore` already knows what is on disk; this
  is a local question and should answer offline, instantly. If it needs the live catalog for the
  size, fall back to the built-in catalog rather than blocking (`ModelCatalog.builtin` is now
  fully pinned, so the fallback is safe).

### Order of work

1. **Resolve the unknown first — do not guess.** What does Core AI report on a device that
   cannot run it? `AIModel.deviceArchitectureName` exists, but its behaviour on an unsupported
   device is unverified here. Find out on hardware before writing the check. **A wrong
   `unsupportedDevice` is worse than no check at all** — it hides the feature on devices that
   work, and nobody reports a feature they never saw.
2. `needsDownload` / `ready` — pure `ModelStore` + catalog lookup, no new knowledge required.
   This alone is most of the value and can ship without step 1.
3. `insufficientStorage` — `URLResourceKey.volumeAvailableCapacityForImportantUsageKey` against
   the resolved size, with headroom (a bundle needs room to unpack, not only to land).
4. `unsupportedDevice` — only once step 1 is answered.
5. Add matching `CoreAIKitError` cases so a *call* fails with the same vocabulary the *query*
   uses. An engineer should never see `notAHuggingFaceRepo` from `CoreAI.summarize`.

### Acceptance

An engineer can write this, and it behaves correctly on a device with no room, a device that is
too old, and a fresh install:

```swift
switch await CoreAI.capability(.transcribe) {
case .ready:                       showTranscribeButton()
case .needsDownload(let bytes):    showDownloadPrompt(bytes)     // "Transcription needs 3.2 GB"
case .insufficientStorage(_, _):   showFreeSpaceMessage()
case .unsupportedDevice:           hideFeatureEntirely()
}
```

---

## The ops, as an app engineer sees them

Every one takes `options: OpOptions = OpOptions()`; none requires naming a model. Sizes are the
first-use download of the default model, from `catalog.json`.

### Text — `qwen3-4b`, 2.5 GB
| Call | Gives you |
|---|---|
| `CoreAI.summarize(text, style:)` | a shorter version |
| `CoreAI.extract(text, as: MyType.self)` | a typed Swift value, schema-constrained (`@Generable`) |
| `CoreAI.translate(text, to:)` | translated text |
| `CoreAI.proofread(text)` | corrected text |

### Speech and audio
| Call | Gives you | Model | Size |
|---|---|---|---|
| `CoreAI.transcribe(audioURL, language:)` | text | `whisper-large-v3-turbo` | 3.2 GB |
| `CoreAI.transcribeMeeting(audioURL)` | speaker-attributed transcript | Sortformer + ASR | + 226 MB |
| `CoreAI.describeAudio(audioURL)` | a description of what is heard | `qwen2.5-omni-3b-audio` | macOS only |
| `CoreAI.speak(text)` | audio | TTS | — |
| `CoreAI.compose(prompt, seconds:)` | music | `stable-audio-open-small` | macOS only |
| `CoreAI.separate(audioURL)` | vocal / instrumental stems | `melband-roformer-vocal` | 470 MB |

### Images and video
| Call | Gives you | Model | Size |
|---|---|---|---|
| `CoreAI.caption(image, style:)` | a caption | `qwen3-vl-2b` | 3.3 GB |
| `CoreAI.detect(in: image)` | `[Detection]` boxes, no NMS | `rf-detr` | **103 MB** |
| `CoreAI.read(documentAt: url)` | markdown, tables preserved | `glm-ocr` | 1.6 GB |
| `CoreAI.upscale(image)` | a larger image | `adcsr-x4` | 1.7 GB |
| `CoreAI.estimateDepth(in: image)` | a depth map | `depth-anything-3-small` | **54 MB** |
| `CoreAI.recognizeAction(videoAt: url)` | ranked action labels | V-JEPA 2 | — |

### Text utilities
| Call | Gives you | Model | Size |
|---|---|---|---|
| `CoreAI.redact(text)` | PII removed | `gliner2-pii` | — |
| `CoreAI.extractEntities(from:labels:)` | zero-shot entities, labels at call time | `gliner2-pii` | — |
| `CoreAI.search(query, in: documents)` | ranked hits | `embeddinggemma-300m` | 1.2 GB |
| `CoreAI.forecast(series)` | 128-step forecast + quantiles | `timesfm-2.5-200m` | 883 MB |

**Two of these are small enough to be uncontroversial** — detection at 103 MB and depth at
54 MB are ordinary app assets, not an ML decision. They are the right first thing to show
someone who has never shipped a model.

---

## How to say it so an app engineer uses it

The audience is not the one currently being reached. Traffic comes from Hugging Face, X and
Qiita — the ML world. Nothing arrives from Swift Forums, Swift Package Index, iOS newsletters or
`r/iOSProgramming`. The product is for iOS engineers and the distribution is for ML engineers.

**Lead with the constraint that disappears, never with the model.** An app engineer does not
want "AI" and has no opinion about Qwen. They want a feature that was previously impossible or
expensive:

> Transcription with no server, no API key, no per-minute cost, and it works on a plane.

That sentence sells. "Whisper large-v3-turbo, int8, token-exact against the fp32 oracle" does
not — it is the reason to *trust* it, which matters on the second read, not the first.

**Rules that follow:**

- **Never open with a model name or a catalog id.** `qwen3.5-2b` means nothing to them.
  `CoreAI.transcribe(url)` means everything.
- **Answer the four shipping questions in the first screen** — device support, download size,
  disk, offline behaviour. An engineer who cannot find the app-size cost assumes it is bad.
- **Lead with the small models.** 54 MB depth estimation is a feature someone ships this
  afternoon. A 3.2 GB transcription download is a product decision that needs a meeting. Put the
  cheap win first and the impressive one second.
- **Drop the ML vocabulary from outbound writing.** Parity gate, oracle, quantization ladder,
  int8hu — all of it is trust-building material for the docs, and all of it is repellent in an
  announcement.
- **Task-shaped titles, not model-shaped.** "Speaker-attributed transcription in an iOS app"
  gets read and searched for. "Sortformer + Parakeet composition" does not.

**Where to say it**: Swift Forums, Swift Package Index, iOS Dev Weekly, `r/iOSProgramming`, and
Apple Developer Forums. Currently zero presence in all of them. The Apple forum is the highest
value of the set because iOS engineers and Apple engineers read the same thread.

**Timing**: iOS 27 GA is the moment this audience can act. There is essentially no public Core
AI question-asking yet — two issues across all of GitHub — because the platform has not
shipped. Whoever has answers written and indexed before that day owns the question afterwards,
and right now nobody is competing for it.
