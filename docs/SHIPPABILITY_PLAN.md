# Plan — the four things between "it runs" and "you can ship it"

The op layer is already shaped for an engineer who has never trained a model. The claim it
makes is stronger than that, though: that adding an on-device feature no longer needs a vendor,
a key, a bill or a privacy review — an entire job removed, not a nicer API over the same job.

That claim is currently false in four places, and each one hands part of the job back. The test
used throughout: **does its absence force the adopter to build the thing we said disappeared?**

Ordered by how much work its absence creates. 1 and 2 decide whether an app can ship at all;
3 and 4 decide whether it is any good.

---

## 1. Background Assets — the download is still the adopter's job

**Now**: no integration. Nothing in `Sources/` references `BackgroundAssets`, `BADownload` or
`BAManager`. First use pulls 1–3 GB from a personal Hugging Face account, inside the app's
lifecycle, and every adopter writes their own progress UI, retry, resumption and
what-if-they-background-the-app.

So "no server" is not yet true. The server is gone; **the download engineering it implied is
not**. It moved to the adopter.

Apple's `BackgroundAssets` exists for exactly this: the system fetches declared assets outside
the app's runtime — including before first launch — and owns retry, roaming, and storage
pressure.

**Build**: a `CoreAIKitAssets` product providing the extension. The adopter declares which ops
their app uses; the kit resolves those to catalog entries and hands the manifest to the system.

- Reuse `CoreAI+Prepare.swift`'s op→model switch. A second mapping will drift.
- Downloads must land in the same `ModelStore` layout the runtime already reads
  (`<repo>/<revision>/<variant>`), so a background-fetched model and a first-use-fetched model
  are indistinguishable afterwards.
- Keep the existing runtime download as the fallback — an app that adopts the extension should
  not break if the system has not fetched yet.
- `CoreAI.onDownload` stays for the fallback path; it is not the primary path any more.

**Done when**: an app that declares `.transcribe` has the model present at first launch without
having written a download screen.

## 2. Model residency — coexistence is currently the adopter's problem

**Now**: `OpModels` (`Sources/CoreAIOps/CoreAI.swift:209`) is an actor holding
`[String: Task<ChatSession, Error>]` and equivalents per kind. **Nothing ever releases them**,
and nothing observes memory pressure.

An app calling `transcribe` then `summarize` keeps two models resident; a third gets it
jetsammed on a phone. Which models can coexist is therefore a question the adopter has to
answer — which is precisely the ML capacity planning we said they would not do.

**Build**: eviction inside `OpModels`.

- LRU across the caches, with a resident-bytes budget derived from
  `os_proc_available_memory()` rather than a constant.
- `DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])` → drop everything
  not currently executing, on `.critical` drop all.
- Eviction must not cancel an in-flight op. An op holding a model pins it until it returns.
- Re-loading after eviction is a cache miss, not an error: the second call is slower, never
  fails.

**Done when**: an app can call every op in sequence on an 8 GB phone without being killed, and
a memory warning during idle frees the resident models.

## 3. Streaming — the escape hatch is currently mandatory

**Now**: no `AsyncStream` anywhere in `Sources/CoreAIOps/`. Every op returns its final result.

A cloud ASR streams partials, and any live transcription or progressive summary needs them. So
the first time an adopter builds a real UX, they must abandon the op layer and drop to the model
layer — the escape hatch becomes the required path rather than an option for people who want
control.

**Build**: streaming variants where partials are meaningful — `transcribe`, and the chat-backed
text ops (`summarize`, `translate`, `proofread`).

```swift
for try await partial in CoreAI.transcribeStream(audioURL) { … }
```

- Same defaults, same `OpOptions`; only the return shape differs.
- Cancellation must be honoured mid-stream: `ChatSession.swift:475` and
  `ConstrainedLoop.swift:52` already check, so the plumbing exists below — verify it survives
  through the op layer.

## 4. `capability(_:)` — the query the kit cannot answer

Full detail in [`CAPABILITY_HANDOFF.md`](CAPABILITY_HANDOFF.md). Summary: before shipping,
an engineer must know whether the device can run it, how large the download is, and whether
there is room. None has an API; only download *progress* does. `prepare()` can be told to act
and cannot be asked whether acting is possible.

**Resolve first, on hardware**: what Core AI reports on a device that cannot run it. A wrong
`unsupportedDevice` hides the feature on devices that work, and nobody reports a feature they
never saw.

---

## What this does not fix

Even with all four, the entry cost is a multi-gigabyte download for most ops. `SnapKit` and
`Lottie` — the packages an engineer adopts in an afternoon without asking anyone — cost
hundreds of kilobytes. Two ops here are in that class (`estimateDepth` 54 MB, `detect` 103 MB);
the rest are a product decision, not an afternoon.

That gap is not closed by better plumbing. It is closed by not needing the download at all when
a smaller answer will do — which is a different piece of work, above this one.
