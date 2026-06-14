# VisualIntel — your own models behind system Visual Intelligence

Point the camera (iOS) or take a screenshot (iPad/Mac), and the results that appear in the
**system Visual Intelligence UI** come from models *you* converted — not Apple's:

- **RF-DETR** (object detection) answers *"what is in this?"*
- **CLIP ViT-B/32** (joint image/text embeddings) answers *"find visually similar photos in my
  library"* — replacing Vision's built-in `GenerateImageFeaturePrintRequest` with your own
  feature print.
- **Qwen3-VL-2B** (vision-language model, via `KitVisionModel`) gives a free-text answer about the
  frame — the **cloud-free counterpart to Visual Intelligence's own "ask,"** which sends your photo
  to ChatGPT on a server. Here the answer is computed by a converted VLM on the device GPU.

RF-DETR/CLIP run through `CoreAIKitVision`; the VLM runs through the kit's VL executor in
`CoreAIKit`. The app does not need to be open: Visual Intelligence launches it in the background to
answer the query.

## Why this works (and the one real risk)

Visual Intelligence integration is **pure App Intents and completely model-agnostic**. The
system hands you a `SemanticContentDescriptor` (it carries `labels` and a
`pixelBuffer: CVReadOnlyPixelBuffer?`) and renders whatever `AppEntity`s you return. There is
**no model parameter, no `FoundationModels` capability, nothing** that inspects what produced the
results. Whatever you run inside `IntentValueQuery.values(for:)` — CLIP, RF-DETR, a VLM, a remote
call — is invisible to the system. So a "third-party model behind Visual Intelligence" is, from
the OS's point of view, just an app returning entities.

Three pieces make the app surface (all in the **main app target — no extension needed**):

| Piece | Type | Role |
|---|---|---|
| Receive pixels, return results | `VisualSearchValueQuery: IntentValueQuery` | runs RF-DETR + CLIP (+ optionally the VLM) on the captured frame |
| Result objects | `DetectedObjectEntity`, `PhotoMatchEntity`, `VisualAnswerEntity` (`@UnionValue VisualSearchResult`) | what the VI UI lists |
| Tap a result | `OpenDetectedObjectIntent`, `OpenPhotoMatchIntent`, `OpenVisualAnswerIntent` (`OpenIntent`) | **required — without an `OpenIntent` per entity type the app never appears in VI** |
| "Continue in app" | `ContinueVisualSearchInAppIntent` (`@AppIntent(schema: .visualIntelligence.semanticContentSearch)`) | optional; opens full results in-app |

The system discovers all of this from the **App Intents metadata extracted at build time** — no
Info.plist key or entitlement is required for the visual-search participation itself.

**The real engineering risk is not surfacing — it is running a Core AI GPU graph inside the
query's out-of-process execution context** (a background app launch with a tighter memory
budget than the foreground app). This example is built to minimize that:

- defaults to **RF-DETR nano** (103 MB, one forward pass);
- **CLIP runs only when photos are indexed**, and only encodes the single incoming frame;
- the photo index is **precomputed** (embeddings + cached thumbnails) at foreground time, so the
  query never touches PhotoKit or rebuilds anything;
- **main-app target, not an App Intents extension** — a background app launch gets the app's
  memory budget and shares the app container, so the persisted index needs no App Group.

## Ask on-device — no cloud (Qwen3-VL), and the memory question it raises

Apple's Visual Intelligence "ask" sends the image to a **server** model (ChatGPT). This example
adds the on-device alternative: your converted **Qwen3-VL-2B** answers on the A19 GPU, offline.

But a ~2 GB VLM collides with the one real risk above. A Visual Intelligence query runs in a
**background launch** of the app, and Apple publishes **no number** for that launch's memory
budget. RF-DETR nano (~100 MB) survives it; whether a 2 GB+ VLM does is the open question, so the
example **measures rather than guesses** and ships **two paths**:

- **A — VLM in the query (experimental, default OFF).** A persisted toggle ("Also answer inside
  Visual Intelligence") makes `values(for:)` also run the VLM in the background launch, logging
  `phys_footprint` + remaining headroom (`os_proc_available_memory`) around the load and inference
  to the unified log. If the process is jetsam-killed, the last breadcrumb pins where it died; the
  RF-DETR/CLIP results still answer, so the surface never goes empty.
- **B — VLM in the foreground (default, robust).** Visual Intelligence shows the light RF-DETR
  teaser; **"Continue in app"** opens the app and the VLM answers there, with the **full app memory
  budget** — sidestepping the background-launch limit entirely.

To measure the background path on device: pick a photo and tap **"Ask the VLM"** once (this
downloads + caches the model and proves the foreground path), flip the toggle on, then trigger
Visual Intelligence. Watch in Console (subsystem `com.coreaikit.visualintel`) whether the
`vlm-load-done` / `vlm-infer-done` stages complete; if the app is killed, the JetsamEvent report
(Settings → Privacy → Analytics Data) names the per-process limit Apple doesn't publish.

## Build

```sh
brew install xcodegen          # if needed
cd Examples/VisualIntel
xcodegen generate
open VisualIntel.xcodeproj
```

Set your signing team (the project defaults to one) and run on an iPhone (camera) or your Mac /
iPad (screenshots). Models download from the Hugging Face Hub on first use and cache on device.

## What the app shows (coach + mirror)

The app itself does nothing magic — the value is in the system. So the screen is built to *say so*
legibly:

- **Identity** — header "Your own model inside Visual Intelligence" + a standing badge row
  `RF-DETR + CLIP · on-device · offline` and "YOUR models, not Apple Intelligence."
- **Coach** — "Use it from the camera or a screenshot," with the trigger steps shown large
  (below). The point: you don't search *in* this app, you trigger the OS.
- **Mirror** — a "Try it" panel that runs the **exact same** `analyze` path the Visual Intelligence
  query uses and draws **RF-DETR boxes over your photo** + a live RF-DETR→CLIP trace + CLIP
  similar-photo thumbnails. Verify the models on a Mac without invoking the system. Tap
  **"Index my photos"** to populate the CLIP index so the similarity surface lights up.
- **Ask on-device** — a card that runs **Qwen3-VL** on the picked photo in the foreground (full
  memory) and shows the answer + latency, plus the toggle that enables the experimental
  background-launch path. This is the foreground control for the memory measurement above.

## Trigger the real Visual Intelligence flow

The demo body is the **system Visual Intelligence UI**, with the app closed:

- **iPhone 16 / 17:** press and hold the **Camera Control** button.
- **Other iPhone:** the **Action button**, or add Visual Intelligence to **Control Center**.
- **Any screenshot:** take a screenshot (iPhone: Side + Volume Up), then tap **Visual
  Intelligence**.
- **iPad / Mac:** take a screenshot → **Search with Visual Intelligence**.

Your RF-DETR detections and CLIP photo matches appear among the system's results; tap one and the
`OpenIntent` launches the app to that detail. A 30-second recording script (camera + screenshot
variants) is in `VIZ_INTEL_STATE.md`.

## Notes / 27-beta caveats

- The `@AppIntent(schema: .visualIntelligence.semanticContentSearch)` macro is the least-stable
  part against the beta SDK. If it fails to compile, the gate (query + entities + `OpenIntent`)
  still stands — comment out `ContinueVisualSearchInAppIntent`.
- App Intents metadata extraction must succeed for discovery (it is on by default for app
  targets). If the app does not appear in VI, confirm the build produced the metadata and that
  each entity has an `OpenIntent`.
- Device build/run is the verification step (the model runs on the GPU/ANE); the gate analysis is
  in `VIZ_INTEL_STATE.md` at the repo root.
