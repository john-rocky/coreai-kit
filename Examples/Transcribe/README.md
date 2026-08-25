# Transcribe — on-device speech-to-text, any ASR model in the catalog

The speech-to-text **model runner**: one app for every `asr` catalog entry — Whisper
large-v3-turbo (100 languages), Qwen3-ASR 1.7B, Parakeet-TDT 0.6B v3. Record a clip, choose a
file, or use the bundled demo; pick the model in the picker; transcription runs fully on
device (first use downloads the model, then it loads from the local cache).

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

## The take-home: `Sources/QuickStart.swift`

One typed function, no UI imports — audio file in, `Transcription` out:

```swift
func transcribe(audio url: URL, model id: String, …) async throws -> Transcription
```

Both shells call exactly this function: the GUI view model is a display shell over it, the
CLI an argument shell. Want transcription in your own app? Copy that file — the glue
(file → 16 kHz mono PCM, mic capture, id → engine dispatch) is all kit API
(`AudioFile.pcm16kMono`, `MicRecorder`, `KitTranscriber`), so it runs as pasted.

The function loads the model per call — the honest end-to-end shape. If your app transcribes
repeatedly, hold the `KitTranscriber` and call `transcribe(samples:)` on it per clip.

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
