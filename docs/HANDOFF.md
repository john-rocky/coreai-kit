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

Existing docs unaffected by any of this: [`GETTING_STARTED.md`](GETTING_STARTED.md),
[`COOKBOOK.md`](COOKBOOK.md), [`STABILITY.md`](STABILITY.md).

## Build order

Ordered by value per unit of work, not by how interesting it is.

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
