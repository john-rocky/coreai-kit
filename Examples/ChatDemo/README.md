# ChatDemo — on-device chat, any chat model in the catalog

The chat **model runner**: one app for every `chat` catalog entry — Qwen3.5/3.6, Gemma 3/4,
LFM2.5, Granite 4, Mistral, MiniCPM5, Nanbeige, … Pick the model in the picker; it downloads
from the Hugging Face Hub on first use, then chats fully on device with live stats
(load / TTFT / tok/s / memory).

```swift
let chat = try await ChatSession(catalog: "qwen3-0.6b")
let reply = try await chat.respond(to: prompt)
```

## Run it

Start with the [0.4.1 installation and tested environment](../../README.md#quickstart).
Run commands below from `Examples/ChatDemo` in the **0.4.1** checkout. Both the
CLI and Xcode app depend on the public **exact 0.4.1** package. Qwen3 0.6B downloads
approximately 352 MB on Mac (456 MB for the iPhone variant).

The initial reply should mention Tokyo. Later requests can vary because sampling is
enabled. The app supports real iPhones, but this release entry is validated on Mac;
an iOS build is not an iPhone runtime check.

GUI (macOS / iOS):

```
open ChatDemo.xcodeproj   # → Run, pick a model in the picker
```

Run the `ChatDemo` scheme on a macOS or iPhone destination (real device — the CoreAI
framework is not in the iOS Simulator SDK). The scheme runs Release — the engine's
per-token host work is ~3x slower in Debug.

CLI (macOS, headless — agents verify with this):

```
swift run -c release chat-cli --prompt "What is the capital of Japan?" --model qwen3-0.6b
swift run -c release chat-cli --list-models
```

The reply goes to stdout; download progress goes to stderr.

## Reproduce the release checks

The public `entry-check` target accepts a mode and a cache directory. Use a new
folder for the first command; retain it for follow-ups. It never deletes the normal
application cache. Each mode exits nonzero on a failed check.

```bash
ENTRY_CACHE="$(mktemp -d)/models"
swift run -c release entry-check chat "$ENTRY_CACHE"
swift run -c release entry-check fm "$ENTRY_CACHE"
swift run -c release entry-check japanese "$ENTRY_CACHE"
swift run -c release entry-check japanese-fm "$ENTRY_CACHE"
```

`chat`, FM and Japanese modes use **0.4.1's built-in Qwen pin**, so a future live-catalog update cannot change that bundle. `fm` checks ORCHID recall on the second turn; the Japanese modes check
streaming output. Results and exact revisions are in the
[release validation record](../../docs/GETTING_STARTED.md#release-041-validation).
`hybrid` is the maintainer's two-turn runtime regression check and downloads a
separate Qwen3.5 0.8B bundle (~1.34 GB). `speak` exercises VoxCPM and writes a WAV.

For the same first-use check in the **physical iPhone app**, select your development
team in Signing & Capabilities, add `COREAI_ENTRY_SMOKE=1` to the scheme's Run
Environment Variables, and run. It uses a new cache folder each time and displays
`PASS: first download and two replies` after recalling ORCHID. The result is also
saved as `Documents/entry-check.txt`. Remove the variable to return to the chat UI.
This is a short entry check, not a performance benchmark or an all-model sweep.

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
