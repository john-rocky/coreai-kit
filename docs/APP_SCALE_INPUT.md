# Design — ops should take the app's unit, not the model's

The speech design ([`SPEECH_API.md`](SPEECH_API.md)) came out of one observation: nothing is
live. Asking the same question of the other domains gives the same answer in a different shape,
and it is worth naming once because it decides most of the remaining work.

**Every op accepts the unit the model accepts. Every app holds a larger unit. The distance
between the two is the job that was supposed to disappear.**

| Domain | Ops accept | The app actually has | What the adopter writes today |
|---|---|---|---|
| Speech | one file, or `[Float]` | a microphone; a two-hour recording | capture loop, chunking, silence detection, partial results |
| Vision | one `CGImage` | a camera; a photo library; a video | frame loop, throttling, orientation, mapping boxes into view space |
| Documents | one image | a multi-page PDF; a scan session | rasterising pages, ordering, stitching output |
| Text | one `String` | a 50-page document; a chat history; a corpus | chunking, map-reduce, context budgeting |

Two sub-problems recur in every row: **live** (results as they arrive) and **scale** (input
larger than the model's window). Speech happens to need both at once, which is why it surfaced
first.

---

## Vision — the camera exists and nothing consumes it

`CoreAIKitVision/CameraFeed.swift` already vends `AsyncStream<CGImage>`, throttled to a target
frame rate. **No op takes it.** `CoreAI.detect(in: image)` is per-frame, so every adopter writes
the loop, decides the throttle, handles device orientation, and converts detection rectangles
into view coordinates — the last of which everyone gets wrong at least once.

```swift
for try await s in CoreAI.watch() {
    s.detections        // [Detection]
    s.image             // the frame they came from
}
```

`watch()` is to the camera what `listen()` is to the microphone: it owns permission, the
session, orientation, and interruption.

**A `Detection` must carry a normalised rect (0…1 in the frame).** Drawing a box over a preview
is then a multiplication, not a coordinate-system puzzle. If detections come back in pixel
coordinates of a rotated buffer, the job was not removed — it was renamed.

Depth has the same shape and the same live need:

```swift
for try await d in CoreAI.watchDepth() { … }     // DepthCamera example does this by hand today
```

## Text — the call should not change, it should stop failing

```swift
let summary = try await CoreAI.summarize(fiftyPageDocument)
```

This is already the code someone writes. It currently sends the whole string at the model and
exceeds its context — there is no chunking anywhere in `Sources/CoreAIOps/`.

The fix adds no API. It makes the existing one true:

- split on structure (paragraphs, headings) rather than a token count, so the pieces are
  summarisable on their own
- summarise pieces, then summarise the summaries, until it fits
- the cost is time, and time is visible, so this is where streaming matters — a progressive
  result is the difference between "working" and "hung"

The same applies to `translate`, `proofread`, `extract` and `redact`: all of them are given a
`String` today and all of them break at length.

**One op has since been built this way, and it is worth reading before the other four are.**
`CoreAI.tidyTranscript` (2026-08-25) chunks because its model leaves no choice: on iOS the
engine caps prompt + generated at 1024 tokens, so a whole transcript does not degrade, it stops
mid-sentence. What that came out as, in `KitTextNormalizer`: cut on **token count at word
boundaries** rather than on structure — raw dictation has no punctuation to split on, which is
the thing being added — pieces **evened out** rather than packed, so a long input never ends in
a 50-token orphan the model rewrites with nothing around it, and `onPartial` after each piece
because the wait is visible. The summarise-the-summaries shape above does not apply: a rewrite
is per-piece and stitches, it does not fold.

## Documents — a PDF is the unit, an image is not

`CoreAI.read(documentAt: url)` takes an image. Nothing in the package references `CGPDF` or a
page count. A real document is a multi-page PDF or a scan session, so the adopter rasterises
pages, runs each, and stitches the markdown — while the op's name implies it already did.

Again: no new API, a truer one.

```swift
let markdown = try await CoreAI.read(documentAt: contractPDF)   // all pages, in order
for try await page in CoreAI.readStream(documentAt: contractPDF) { … }  // long documents
```

## What this means for the shape of the work

Most of it is **not new API surface**. Three of the four rows are existing calls learning to
accept the real unit — which is the better kind of change, because nothing has to be discovered
or documented for an adopter to benefit.

New verbs are needed only where the input is genuinely live and has no representation today:

| New | Why |
|---|---|
| `CoreAI.listen()` | the microphone has no op |
| `CoreAI.watch()` | the camera has a stream and no op |
| `*Stream` variants | progress on inputs large enough that the wait is visible |

And one requirement that is not an API at all: **normalised coordinates on `Detection`**, because
the alternative hands a coordinate-system problem back to the person we told would not need one.

## Ordering

1. **Text at scale** — chunking for `summarize` / `translate` / `proofread` / `extract`. No new
   API, no new models, no device work, and it fixes calls that fail today.
2. **`watch()`** — the stream already exists; this is wiring plus the coordinate contract.
3. **`listen()` + VAD** — the largest demand, but it needs a new component and the iOS ports to
   be affordable (see [`SPEECH_API.md`](SPEECH_API.md)).
4. **PDF** — smallest audience of the four, but the cheapest once page handling exists.

Nothing here needs a new model. Every part of it is the distance between what the models take
and what apps hold.
