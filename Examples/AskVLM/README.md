# Qwen3-VL (AskVLM) — your VLM as its own Visual Intelligence tab

Apple's Visual Intelligence "ask" sends your photo to a server model (ChatGPT). This app answers
with your converted **Qwen3-VL-2B** on the device GPU instead — offline, no cloud. Because Visual
Intelligence shows **one tab per app**, this dedicated app surfaces as its own **"Qwen3-VL"** tab
next to Google (and next to the `VisualIntel` example's "RF-DETR" detection tab).

This split is deliberate: detection is instant, but a VLM takes time. Keeping the VLM in its own
tab means the fast detection tab never waits for it — and the system shows its own loading state in
the Qwen3-VL tab while the model thinks, then renders the answer.

## How it works

The whole Visual Intelligence surface is one `IntentValueQuery` (in the **main app target**, no
extension): it receives the captured pixels, runs `KitVisionModel` (the kit's VL executor) on the
whole frame, and returns a single answer entity. Each entity needs an `OpenIntent` or the app never
appears in Visual Intelligence. The VLM runs through `CoreAIKit`; the system never sees the model.

## The one real constraint: warm it first

A Visual Intelligence query runs in a **background launch** of the app, which has **no bandwidth to
download** the ~2.3 GB model and a **query timeout** the first cold compile (~30 s) can exceed. So:

1. Open the app, pick a photo, and **"Ask the VLM"** once — this downloads + warms the model
   (cached persistently in Application Support).
2. Then trigger Visual Intelligence → open the **Qwen3-VL** tab. Warm, the answer comes in ~1.4 s.

The on-device measurement (`phys_footprint` ~440 MB, `os_proc_available_memory`) is logged to the
unified log (subsystem `com.coreaikit.askvlm`); the 2 GB VLM survives the background launch.

## Build

```sh
brew install xcodegen          # if needed
cd Examples/AskVLM
xcodegen generate
open AskVLM.xcodeproj
```

Run on an iPhone (camera) or Mac / iPad (screenshots). The model downloads from the Hugging Face
Hub on the first "Ask the VLM" and caches on device.
