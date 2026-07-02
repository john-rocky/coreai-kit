# VLChat — the vision-language runner (GUI + CLI, any `vlm` catalog id)

Pick a photo, ask about it, and a **vision-language model running entirely on your device**
answers. No network, no cloud — the image never leaves the device.

The take-home is [`Sources/QuickStart.swift`](Sources/QuickStart.swift): one typed function
(image + prompt → answer), no UI. The CLI is an argument shell over it; the GUI drives the
same `KitVisionModel(catalog:)` gesture behind a FoundationModels `LanguageModelSession`:

```swift
let vlm = try await KitVisionModel(catalog: "qwen3-vl-2b")   // downloads decoder + vision tower
let session = LanguageModelSession(model: vlm)
let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
let answer = try await session.respond(to: Prompt {
    "What is in this photo?"
    Attachment(image.cgImage, orientation: image.orientation)
})
```

Under the hood `KitVisionModel`/`KitVisionExecutor`/`VLRuntime`:

- create the decoder engine with the four **static multimodal inputs** the VL bundle declares
  (`image_embeds`, `deepstack_embeds`, `rope_shift_start`, `rope_shift_amount`) — the stock
  load path can't, which is why a generic `KitLanguageModel` rejects a VL bundle;
- run the paired vision tower (a plain `.aimodel`) through CoreAIKit's `GraphModel` and write
  the embeds into those buffers;
- splice the vision block into the prompt and rewrite the image-pad ids to the decoder's
  extension id space, so the picked photo's pixels actually reach the model.

## Models

The picker lists every `vlm` entry in the catalog
(`ModelCatalog.builtin.available(.vlm)`): Qwen3-VL 2B / 4B (iPhone + Mac) and
8B (Mac-only — its decoder exceeds the iPhone jetsam ceiling). Downloaded on first
load and cached.

## Build & run

```bash
open VLChat.xcodeproj   # run on My Mac, or on an iPhone (iOS 27)

# agents / headless (macOS):
swift run vlchat-cli --model qwen3-vl-2b --image sample.jpg --prompt "What is in this image?"
```

Run the GUI in **Release** — the engine's per-token host work is ~3× slower unoptimized.

## Notes

- One image per conversation (the latest photo you pick). Picking a new photo starts a fresh
  chat about it.
- The VL path streams the answer; Qwen3-VL is a thinking model, but the Instruct weights
  answer image questions directly.
- Tool calling / guided generation are not wired for the VL model — use the text
  `KitLanguageModel` for those.
