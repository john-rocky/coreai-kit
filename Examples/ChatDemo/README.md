# ChatDemo — on-device chat, any chat model in the catalog

The chat **model runner**: one app for every `chat` catalog entry — Qwen3.5/3.6, Gemma 3/4,
LFM2.5, Granite 4, Mistral, MiniCPM5, Nanbeige, … Pick the model in the picker; it downloads
from the Hugging Face Hub on first use, then chats fully on device with live stats
(load / TTFT / tok/s / memory).

```swift
let chat = try await ChatSession(catalog: "qwen3.5-2b")
let reply = try await chat.respond(to: prompt)
```

## Run it

GUI (macOS / iOS):

```
open ChatDemo.xcodeproj   # → Run, pick a model in the picker
```

Run the `ChatDemo` scheme on a macOS or iPhone destination (real device — the CoreAI
framework is not in the iOS Simulator SDK). The scheme runs Release — the engine's
per-token host work is ~3x slower in Debug.

CLI (macOS, headless — agents verify with this):

```
swift run chat-cli --prompt "What can you do, offline?" --model qwen3.5-2b
swift run chat-cli --list-models
```

The reply goes to stdout; download progress goes to stderr.

## The take-home: `Sources/QuickStart.swift`

One typed function, no UI imports — prompt in, reply out:

```swift
func ask(_ prompt: String, model id: String, …) async throws -> String
```

The CLI is an argument shell over exactly this function; the GUI drives the same
`ChatSession(catalog:)` gesture, held across turns for its transcript. Want an on-device LLM
in your own app? Copy that file — the catalog id resolves the repo, platform variant, and
engine hint internally, so it runs as pasted.

The function loads the model per call — the honest end-to-end shape. Multi-turn? Hold the
`ChatSession` and call `respond(to:)` per turn (it keeps the conversation history);
`streamResponse(to:)` yields tokens as they decode.

## Integration checklist

- SPM: `github.com/john-rocky/coreai-kit` → product `CoreAIKit`
- Info.plist: none needed
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit`
- First run downloads the model → cached in `Application Support/CoreAIKit/Models`
  (progress callback). On iOS, only models with a published `ios/` variant appear in the picker.
- Measure in Release — Debug is ~3× slower on per-token host work

## Benchmarking (FALCON_BENCH)

Launching the app with the `FALCON_BENCH` environment variable set runs a headless on-device
benchmark of a local bundle pushed to `Documents/falcon-bench/` (TTFT / decode tok/s /
footprint → stdout + `result.txt`), then exits. `FALCON_ENGINE` overrides the engine variant.
