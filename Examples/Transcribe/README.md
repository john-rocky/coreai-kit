# Transcribe — on-device speech-to-text, any ASR model in the catalog

The speech-to-text **model runner**: one app for every `asr` catalog entry — Whisper
large-v3-turbo (100 languages), Qwen3-ASR 1.7B, Parakeet-TDT 0.6B v3, Fun-ASR-Nano 2512. Pick
the model in the picker (it loads once and stays loaded), record a clip, choose a file, or use
the bundled demo; transcription runs fully on device (first use downloads the model, then it
loads from the local cache). Under the transcript, the app's own measurement of the run: the
clip's length and how long the transcription call took, the model load not included.

```swift
let transcriber = try await KitTranscriber(catalog: "whisper-large-v3-turbo")
let samples = try AudioFile.pcm16kMono(url)  // any wav/m4a/mp3 → 16 kHz mono Float
let result = try await transcriber.transcribe(samples: samples)
// result.text, result.language
```

## Run it

GUI (macOS / iOS):

```
open Transcribe.xcodeproj   # → Run, pick a model in the picker
```

CLI (macOS, headless — agents verify with this):

```
swift run transcribe-cli --audio sample.wav --model whisper-large-v3-turbo
swift run transcribe-cli --list-models
```

The transcript goes to stdout; download progress and the detected language go to stderr.

Hotwords — names and terms the model should spell right — go to Fun-ASR-Nano, comma-separated
(in the app, the Hotwords field that appears when it is picked):

```
swift run transcribe-cli --audio memo.wav --model fun-asr-nano-2512 --hotwords "Tavenmoor, Anwen Thorsby"
```

Hands-off, for a recording or a smoke run: `Transcribe.app/Contents/MacOS/Transcribe -autoplay 1
-model fun-asr-nano-2512 -clip <wav> -hotwords "…" -play 1 -log 1` loads the model, plays the
clip, then presses Transcribe with the Hotwords field empty and again with it filled
(`Sources/Autoplay.swift` has every flag).

## The take-home: `Sources/QuickStart.swift`

No UI imports — a model you keep, and a one-call function over it:

```swift
enum SpeechModel { init(catalog id: String, …) async throws; func transcribe(samples:hotwords:…) }
func transcribe(audio url: URL, model id: String, hotwords: [String] = [], …) async throws -> Transcription
```

Both shells go through this file: the GUI view model holds a `SpeechModel` (loaded when the
model is picked, kept between clips), the CLI calls `transcribe(audio:model:)`, which loads the
model per call. `SpeechModel` is `KitTranscriber` for every id except Fun-ASR-Nano, held as
`KitFunASRModel` because only it takes hotwords. Want transcription in your own app? Copy that
file — the glue (file → 16 kHz mono PCM, mic capture, id → engine dispatch) is all kit API
(`AudioFile.pcm16kMono`, `MicRecorder`, `KitTranscriber`, `KitFunASRModel`), so it runs as
pasted.

## Integration checklist

- SPM: `github.com/john-rocky/coreai-kit` → product `CoreAIKit`
- Info.plist: `NSMicrophoneUsageDescription` (only if you record; the permission prompt and
  record button are app chrome — `MicRecorder` does the capture)
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit` for Whisper
  (1.6 GB fp16 graph; 3.2 GB AOT bundle on iPhone)
- First run downloads the model → cached in Application Support (progress callback)
- Measure in Release — Debug is ~3× slower on per-token host work

## Models

| Catalog id | Model | Platforms | Notes |
|---|---|---|---|
| `whisper-large-v3-turbo` | Whisper large-v3-turbo | macOS + iOS | 100 languages, auto-detect, ≤30 s window |
| `qwen3-asr-1.7b` | Qwen3-ASR 1.7B | macOS | LLM-engine ASR, 52 languages |
| `parakeet-tdt-0.6b-v3` | Parakeet-TDT 0.6B v3 | macOS | TDT transducer, fastest of the three. Apple ships this model itself since `coreai-models` #136 (08-07), with live streaming in #184 — on iOS too. Reach for this entry when you want it inside the kit's one-call ASR surface, not because it is otherwise unavailable |
| `fun-asr-nano-2512` | Fun-ASR-Nano 2512 | macOS + iOS | zh / en / ja, hotword list in the prompt |
