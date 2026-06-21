# AudioChat — on-device audio *understanding* (macOS)

A SwiftUI demo of **audio understanding** (not transcription) running entirely on-device through
[CoreAIKit](../../README.md): load a clip, ask *"what do you hear?"*, and a local
**Qwen2.5-Omni Thinker** describes the sounds — *"I hear a loud, continuous hissing sound."*,
*"…a continuous sine wave sound."*, *"…a series of beeps."*

```swift
let model = try await KitAudioModel(
    decoderBundleAt: decoderBundle, encoderModelAt: encoderModel, arch: .qwen2_5Omni3B)
try await model.attach(samples: pcm16kMono)          // mel → audio encoder → static buffer
let answer = try await LanguageModelSession(model: model).respond(to: "What do you hear?")
```

## How it works

Two Core AI bundles + a Swift front end, wired exactly like the kit's VL path:

- **Audio encoder** (`*_audio_encoder_fp16_k15.aimodel`, 1.2 GB) — a fixed-shape Whisper-style
  tower run once per clip through `CoreAIKitVision.GraphModel`.
- **Text decoder** (`*_thinker_int8lin_n750_s1`, 3.9 GB) — a Qwen2.5-3B decoder on the Core AI
  **pipelined engine**; the audio embeddings ride **one static-input buffer** (`audio_embeds`),
  and the prompt's `<|AUDIO|>` placeholders carry extension ids `vocab + slot` the graph gathers.
- **Mel front end** — Whisper-large-v3 log-mel in Accelerate/vDSP (`AudioMelPreprocessor`),
  bit-exact with the HF feature extractor (gated cos 1.0).

FoundationModels has no audio attachment, so the clip is attached to the runtime out-of-band
(`model.attach(samples:)`) before the session asks the question; the executor renders the audio
block from the attached count. v1: one clip per session.

## Run (macOS)

```sh
cd Examples/AudioChat
xcodegen generate
open AudioChat.xcodeproj   # build & run the AudioChat scheme (Release)
```

The view model points at local bundles under
`coreai/ondevice/artifacts/` (the conversion repo's output) — edit `AudioChatModel` if yours live
elsewhere. **Load model** → **Choose audio…** (or **Demo: white noise**) → **Ask**.

Audio understanding is **Mac-only** for now (the 3.9 GB decoder is GPU-pipelined; iPhone is a
follow-up once the decoder gets its ANE static-shape rework). Any audio file is decoded to
16 kHz mono and trimmed to ≈30 s.
