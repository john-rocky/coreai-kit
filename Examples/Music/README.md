# Music — text-to-music generation on device

A prompt in, audio out, fully on device. "128 BPM tech house drum loop" becomes a wav you can
play or drop into a project. The model is
[Stable Audio Open Small](https://john-rocky.github.io/coreai-model-zoo/models/stable-audio-open-small/),
which generates faster than real time on an iPhone.

The whole ML surface is one call:

```swift
let musician = try await KitMusician(catalog: "stable-audio-open-small")
let audio = try await musician.compose(prompt, seconds: 11)
```

`Sources/QuickStart.swift` is the take-home version of that — prompt in, `SpokenAudio` out, no
UI — and both the app and the CLI are shells over it.

## Run

```bash
swift run music-cli --prompt "128 BPM tech house drum loop" --output loop.wav

xcodegen generate                      # app
open Music.xcodeproj
```

First use downloads the model from the Hugging Face Hub; later runs load from the local cache.

## Notes

- `--seconds` sets the clip length. The model is trained for short clips — loops, stings, beds —
  not for songs with structure, so judge it on that job.
- Any text-to-music model in the catalog works: `ModelCatalog.builtin.available(.music)` lists
  them, and `--model <catalog-id>` swaps the one the CLI uses.
