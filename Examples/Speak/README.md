# Speak — the text-to-speech runner (GUI + CLI, any `tts` catalog id)

Type a sentence, an on-device TTS model speaks it. Everything (LM, diffusion, vocoder)
runs locally — the text never leaves the device.

The take-home is [`Sources/QuickStart.swift`](Sources/QuickStart.swift): one typed function
(text → `SpokenAudio`), no UI. The CLI is an argument shell over it; the GUI drives the same
`KitSpeaker(catalog:)` and plays the result:

```swift
let speaker = try await KitSpeaker(catalog: "voxcpm-0.5b")
let audio = try await speaker.synthesize("Hello from Core AI.")
// audio.samples: 16 kHz mono PCM in [-1, 1]
```

## Build & run

```bash
open Speak.xcodeproj   # run on My Mac, or on an iPhone (iOS 27)

# agents / headless (macOS):
swift run speak-cli --text "Hello from Core AI." --output hello.wav
```

Streaming? `KitSpeaker.synthesizeStreaming(_:onChunk:)` hands you ~0.5 s chunks as they
decode, so playback can start before the whole clip exists.
