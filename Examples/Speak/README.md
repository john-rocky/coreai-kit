# Speak — the text-to-speech runner (GUI + CLI, any `tts` catalog id)

Type a sentence, an on-device TTS model speaks it. Everything (LM, diffusion, vocoder)
runs locally — the text never leaves the device.

The take-home is [`Sources/QuickStart.swift`](Sources/QuickStart.swift): one typed function
(text → `SpokenAudio`), no UI. The CLI is an argument shell over it; the GUI drives the same
`KitSpeaker(catalog:)` and plays the result:

```swift
import CoreAIKit

let speaker = try await KitSpeaker(catalog: "voxcpm-0.5b")
let audio = try await speaker.synthesize("Hello from Core AI.")
// audio.samples: 16 kHz mono PCM in [-1, 1]
```

## Build & run

Start with the [0.4.1 installation and tested environment](../../README.md#quickstart).
Use the **0.4.1** checkout: the CLI and Xcode app depend on the public exact **0.4.1**
package. The release demo uses VoxCPM 0.5B and a fixed voice on Mac.

```bash
open Speak.xcodeproj   # run on My Mac, or a real iPhone

# agents / headless (macOS):
swift run -c release speak-cli --model voxcpm-0.5b --text "Hello from Core AI." --output hello.wav
open hello.wav
```

Run these commands from `Examples/Speak`, with the OS and Xcode versions in the
[package requirements](../../README.md). The model, tokenizer and host tables download
on first use (~1.37 GB on Mac, ~1.68 GB on iPhone, per the catalog) and cache in
`Application Support/CoreAIKit/Models`. No model files belong in the app bundle.

Streaming? `KitSpeaker.synthesizeStreaming(_:onChunk:)` hands you ~0.5 s chunks as they
decode, so playback can start before the whole clip exists.
