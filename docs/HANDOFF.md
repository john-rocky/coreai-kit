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
| [`TASK_MAP.md`](TASK_MAP.md) | Which capabilities are worth having at all — measured demand, what Apple already ships, what only this kit has | Analysis; **speech rows corrected 2026-08-05 from the SDK** |
| [`APP_SCALE_INPUT.md`](APP_SCALE_INPUT.md) | The one pattern behind most of the remaining work: ops take the model's unit, apps hold a larger one | Analysis |
| [`SHIPPABILITY_PLAN.md`](SHIPPABILITY_PLAN.md) | The four gaps between "it runs" and "you can ship it" | Plan, ready to build |
| [`CAPABILITY_HANDOFF.md`](CAPABILITY_HANDOFF.md) | One of those four in detail — `CoreAI.capability(_:)`, plus the full ops list and how to describe them | **Built 2026-08-11**, except the device-age case |
| [`SPEECH_API.md`](SPEECH_API.md) | The speech design — `listen()`, streaming, diarization back-fill | Design; **read the state block above first** — Apple ships the endpointer and the streaming transcriber, so most of this shrank |
| [`BACKEND_ROUTING.md`](BACKEND_ROUTING.md) | Ops picking Apple's free backend when it suffices, the catalog model when it does not — the largest lever on entry cost | Design, gated on measurement |
| [`EVERYWHERE_CONCEPT.md`](EVERYWHERE_CONCEPT.md) | Where routing leads: one API where every on-device AI task has an answer, plus the measured dependency structure and the light path it unlocks | Product concept |

Existing docs unaffected by any of this: [`GETTING_STARTED.md`](GETTING_STARTED.md),
[`COOKBOOK.md`](COOKBOOK.md), [`STABILITY.md`](STABILITY.md).

## Build order

Ordered by value per unit of work, not by how interesting it is.

> ## State as of 2026-08-11
>
> Built, tested and pushed on branch `live-pipeline` (9 commits, 77 tests, iOS builds):
> steps **4** (model residency) and **5** (`watch()`), plus work this plan did not have —
> `LivePipeline` as a *shared* primitive with a thermal governor, video ops
> (`CoreAI.scan`), `KitTracker`, `VoiceActivityDetector`, `capability()`, `coreai-doctor`,
> and the transcription default moving to Apple's backend. Steps **1, 2, 3, 6, 7 are still
> as written below**, with the corrections in the next paragraph.
>
> **What changed in the plan itself.** `TASK_MAP.md` and `SPEECH_API.md` were both written
> before anyone read the iOS 27 `Speech.framework` interface. It ships `SpeechAnalyzer` +
> `SpeechTranscriber` (streaming and word timestamps, both listed here as kit advantages)
> and `SpeechDetector`, which is an endpointer — the component `SPEECH_API.md` calls "the
> only genuinely new component" behind `listen()`. So **step 6 shrank**: `listen()` is now
> mostly wiring Apple's modules, and the kit's speech value is diarization plus voice
> cloning. The iOS Parakeet/Kokoro ports that step 6 lists as a precondition were
> **dropped** — they made a free capability 1.3 GB cheaper.
>
> **Open, deliberately deferred** (owner decided 2026-08-04): the coordinate contract. Boxes
> are normalised and that is not enough — the preview runs at the session preset's aspect
> under `.resizeAspectFill` while the model is fed a 3:4 buffer, so drawing needs the
> aspect-fill correction `Examples/DetectCamera`'s overlay does by hand. `TODO` at
> `Sources/CoreAIKitCore/LivePipeline.swift` (`LiveResult` needs a frame size) and
> `Examples/LiveCamera/Sources/CameraPreview.swift`. **This part of step 5 is not done.**
>
> **Never measured.** No device numbers for `LivePipeline` or `KitTracker` — `DeviceGate`
> in `Examples/LiveCamera` was written for exactly this and has not been run to completion.
> The 33–39 FPS in the README belongs to `DetectCamera`'s hand-rolled loop and must not be
> quoted for `LivePipeline`. Power is still unmeasured, as the precondition below says.
>
> **Also never measured (2026-08-25): `tidyTranscript` on a phone.** The S1-mini *bundle* is
> device-verified — 276/276 + 27/27 token-exact vs the Mac, and the 1024-token ceiling was
> measured there. The *op path over it* was only run on a Mac. So `KitTextNormalizer`'s
> `contextTokens = 1024` and its 450-token chunk budget are derived from the engine source
> plus that evidence, not observed: nobody has watched a chunk complete on iOS, or checked
> that the stitched result matches the Mac's. Nothing in the docs claims otherwise — keep it
> that way. `Examples/Tidy` builds for iOS and is what to run.
>
> **Still unresolved, and it blocks `capability()`'s last case.** What Core AI reports on
> hardware too old to run a bundle. `capability()` returns `unsupportedDevice` only for
> facts knowable without a device (id absent from the catalog, model not published for the
> platform, locale Apple cannot do). The age check is not implemented, per the precondition.
>
> **Try it on a device: `Examples/LiveCamera`.** Four tabs, one per live task, `-gate` for a
> transcript instead of taps, `swift run live-cli` for the offline half with no device.

0. **`CoreAI.transcribe(_:normalize:)`** — chain ASR into the text normalizer in one call.
   `CoreAI.tidyTranscript` shipped 2026-08-25 and is the real product shape's second half;
   the first half is that nobody wants to call two ops. Deliberately deferred until the op
   was proven on its own, and it stays deferred until two things are answered rather than
   assumed: **two models resident at once** (the ASR model plus 796 MB, on a phone, which
   is what `ModelResidency` exists to arbitrate), and **what the flag means with no
   `options.model`** — that path goes to the *system* transcriber, so the composed call
   spans an Apple backend and a catalog model. Cheapest of anything on this list, and the
   one with the most ways to be quietly wrong.

1. **Text at scale** — chunking for `summarize` / `translate` / `proofread` / `extract`.
   No new API, no new model, no device. It makes calls that fail today stop failing.
   `KitTextNormalizer` is the worked example — see `APP_SCALE_INPUT.md` §Text for what its
   chunker settled on and why structure-splitting is not it.
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
- ~~**Speech is not affordable on iPhone yet.**~~ **Resolved 2026-08-05, by deleting the
  problem rather than solving it.** The iOS Parakeet/Kokoro exports this called a precondition
  were dropped: they would have made a capability Apple ships for free 1.3 GB cheaper.
  `transcribe` routes to `SpeechAnalyzer` (0 bytes) and `transcribeMeeting` is Sortformer plus
  Apple's transcriber at **238 MB**, measured. The 4.3 GB figure was also built on catalog
  sizes that were wrong; see the size-measurement commit.
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
