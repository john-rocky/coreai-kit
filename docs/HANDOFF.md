# Handoff — start here

Everything designed on 2026-07-28/29, in the order it should be read and built. Written for a
session that will implement it and was not present for the reasoning.

**The goal**: an iOS engineer who has never trained a model can add an on-device feature to a
shipping app without a vendor, a key, a bill, or a privacy review — an entire job removed, not a
nicer API over the same job.

**Where that claim is currently false** is what these documents are about.

---

## The documents

| Read | For | State |
|---|---|---|
| [`TASK_MAP.md`](TASK_MAP.md) | Which capabilities are worth having at all — measured demand, what Apple already ships, what only this kit has | Analysis, evidence attached |
| [`APP_SCALE_INPUT.md`](APP_SCALE_INPUT.md) | The one pattern behind most of the remaining work: ops take the model's unit, apps hold a larger one | Analysis |
| [`SHIPPABILITY_PLAN.md`](SHIPPABILITY_PLAN.md) | The four gaps between "it runs" and "you can ship it" | Plan, ready to build |
| [`CAPABILITY_HANDOFF.md`](CAPABILITY_HANDOFF.md) | One of those four in detail — `CoreAI.capability(_:)`, plus the full ops list and how to describe them | Plan, ready to build |
| [`SPEECH_API.md`](SPEECH_API.md) | The speech design — `listen()`, streaming, diarization back-fill | Design, has a precondition |
| [`BACKEND_ROUTING.md`](BACKEND_ROUTING.md) | Ops picking Apple's free backend when it suffices, the catalog model when it does not — the largest lever on entry cost | Design, gated on measurement |
| [`EVERYWHERE_CONCEPT.md`](EVERYWHERE_CONCEPT.md) | Where routing leads: one API where every on-device AI task has an answer, plus the measured dependency structure and the light path it unlocks | Product concept |

Existing docs unaffected by any of this: [`GETTING_STARTED.md`](GETTING_STARTED.md),
[`COOKBOOK.md`](COOKBOOK.md), [`STABILITY.md`](STABILITY.md).

## Build order

Ordered by value per unit of work, not by how interesting it is.

> **State, 2026-08-03.** Steps 4 and 5 are built and tested (`Sources/CoreAIOps/ModelResidency.swift`,
> `Sources/CoreAIKitCore/LivePipeline.swift`, `Sources/CoreAIKitVision/LiveVision.swift`,
> `Sources/CoreAIOps/CoreAI+Watch.swift`; 28 tests in total with the video work below).
> Two things the plan did not have: the live loop is a **shared** primitive rather than a
> vision-only one — `listen()` is meant to ride the same `LivePipeline` — and it carries a
> **thermal governor**, because the "nothing here has been measured for power" warning under
> *Preconditions* applies to exactly the pipeline step 5 introduces. Also settled while
> building: `Detection` already carried a normalised rect, so the coordinate contract step 5
> asks for was never a change. Steps 1, 2, 3, 6 and 7 are still as written.
>
> **Video is now a row on the map.** `APP_SCALE_INPUT.md`'s table has four rows — speech,
> vision, documents, text — and no video, which turned out to be the only area with no design
> at all rather than a design not yet built. `CoreAI.scan(videoAt:)` and `VideoFile.stream`
> fill it. The load-bearing decision there is measured, not chosen: seeking and sequential
> decode cross over near one sample per second of 30 fps source, and picking one and standing
> by it costs 17× at one end of the range and 4× at the other.
>
> **Live drops, offline does not.** Worth carrying into the speech work: `watch()` throws
> frames away when the model falls behind, `scan()` delivers every sample it was asked for.
> `listen()` is the live policy; `transcribeStream(file:)` is the offline one, and they are
> not the same pipeline configuration.
>
> **Try it on a device: `Examples/LiveCamera`.** Four tabs, one per live task, with the
> measured stats and the governor on screen, plus `swift run live-cli` for the offline half.
> Launch it with `-gate` for a device transcript instead of taps (`DeviceGate`).
>
> **Known open, deliberately deferred — the coordinate contract.** Step 5 above asks for
> normalised rects so that drawing a box is a multiplication. `Detection` has them, and it is
> still not enough: the preview runs at the session preset's aspect under `.resizeAspectFill`
> while the model is fed `LiveVision.captureSize(forModelInput:)`, so boxes drawn without the
> aspect-fill correction are visibly offset — confirmed on an iPhone 17 Pro on 2026-08-03.
> The fix is a frame size on `LiveResult` plus the mapping shipped in `CoreAIKitUI` instead of
> re-derived per app; `Examples/DetectCamera`'s `DetectionOverlay` is the working version to
> lift. Marked `TODO` at `Sources/CoreAIKitCore/LivePipeline.swift` (`LiveResult`) and
> `Examples/LiveCamera/Sources/CameraPreview.swift` (`DetectionOverlay`). Until it lands,
> **this part of step 5 is not done**, whatever the rest of the row says.
>
> **Not measured yet.** No device numbers for this path — the gate has not been run to
> completion (the phone dropped off mid-session). The 33–39 FPS in the README belongs to
> `DetectCamera`'s hand-rolled loop, not to `LivePipeline`, and must not be quoted for it.

1. **Text at scale** — chunking for `summarize` / `translate` / `proofread` / `extract`.
   No new API, no new model, no device. It makes calls that fail today stop failing.
   *(`APP_SCALE_INPUT.md` §Text)*
2. **`capability(_:)`** — the query the kit cannot be asked. Most of the value lands without
   solving the device-support question; do the rest of it first and that part last.
   *(`CAPABILITY_HANDOFF.md`)*
3. **Background Assets** — the largest single job still handed back to the adopter. Nothing in
   `Sources/` references it, so every app that ships this writes its own download UX.
   *(`SHIPPABILITY_PLAN.md` §1)*
4. **Model eviction** — `OpModels` caches every load and never releases; three ops in sequence
   gets an app killed on a phone. *(`SHIPPABILITY_PLAN.md` §2)*
5. **`watch()`** — `CameraFeed` already vends `AsyncStream<CGImage>` and no op consumes it. Wiring
   plus one contract: `Detection` must carry a normalised rect.
   *(`APP_SCALE_INPUT.md` §Vision)*
6. **`listen()` + VAD** — the largest demand on the map, but see the precondition below.
   *(`SPEECH_API.md`)*
7. **Streaming variants and PDF** — smallest audiences, cheapest once the above exists.

Running beside all of it, on its own clock: **[`BACKEND_ROUTING.md`](BACKEND_ROUTING.md)**. Its
first two steps (result types carrying `.backend`, and one measured Vision-vs-GLM-OCR
comparison) can start immediately and gate everything after them. It moves the entry cost of
most ops from gigabytes to zero, which is the number that decides whether any of the rest gets
adopted.

## Preconditions and unknowns — resolve before writing the code they gate

- **Device support is unverified.** What Core AI reports on a device that cannot run it is not
  known here. `AIModel.deviceArchitectureName` exists; its behaviour on unsupported hardware does
  not. Find out on a device before implementing `unsupportedDevice`. **A wrong answer hides the
  feature on devices that work, and nobody reports a feature they never saw.**
- **Speech is not affordable on iPhone yet.** The stack is 4.3 GB because Parakeet (1.3 GB) and
  Kokoro (341 MB) are macOS-only; with iOS variants published it is 2.1 GB. Both are already
  converted — only the iOS export is missing. Treat this as a precondition for the speech work,
  not a follow-up.
- **Nothing here has been measured for power.** A live audio or camera pipeline runs the GPU
  continuously. A feature that drains a phone in an hour gets removed by whoever ships it, and no
  number on this map addresses that.

## Two decisions already made, so they are not re-litigated

**Speech gets its own product, not its own repository.** It is the largest demand, the deepest
inventory here, and contains the one capability Apple has no answer for. But one maintainer with
two repositories means two catalogs and two CIs — and on 2026-07-28 two pieces of split state in
this project were found to have silently drifted apart. One repo, one catalog, one CI; a
`CoreAISpeech` SwiftPM product so an adopter links only what they need. The line for that product
is the human voice: recognition, synthesis, diarization, and their compositions. Music generation
and source separation are audio but not speech, have no demand signal, and stay in the general
kit.

**Examples teach patterns, not verticals.** [`Examples/ScanToType`](../Examples/ScanToType) is
the reference for the shape: a generic core, one concrete type as an instance, and the swap made
explicit in the README. A vertical example makes the kit look narrower than it is and commits a
solo maintainer to a domain there is no reason to own.

## How to describe any of this to the people it is for

The full version is in [`CAPABILITY_HANDOFF.md`](CAPABILITY_HANDOFF.md) §"How to say it". The
short version: **lead with the constraint that disappears, never with the model name.** An app
engineer has no opinion about which model it is and every opinion about whether they need a
backend, a key, a bill and a conversation with legal. "3 lines of Swift" is not the pitch; "no
server, no key, no per-minute cost, works on a plane" is.

And put the small models first. `estimateDepth` at 54 MB and `detect` at 36–103 MB are ordinary
app assets someone adds on a Tuesday. A 3.2 GB transcription download is a product decision that
needs a meeting. Leading with the impressive one loses the person who would have shipped the
cheap one.
