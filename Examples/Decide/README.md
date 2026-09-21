# Decide — typed decisions on device

A state and a typed question in, an answer with its probability out — nothing generated. The
model scores one prompt and the answer is read at its answer slot: **choice** (which of these
options), **score** (where on this scale), **noul** (yes or no, as P(yes)). One line:

```swift
let a = try await CoreAI.decide(
    text,
    ["reply":  .noul("Does the speaker want a response from the assistant?"),
     "topic":  .choice("What is the message about?", ["billing", "delivery", "other"]),
     "urgent": .score("How urgent is it?", levels: ["can wait", "this week", "today"])])
a["reply"]?.noul      // P(yes)
a["topic"]?.choice    // "delivery"
a["urgent"]?.score    // expected level, 0…2
```

Three screens, one loaded model, every decision with its measured milliseconds:

| Screen | What it decides | Shape |
|---|---|---|
| **Speech gate** | Record, split into utterances, transcribe each on device (Apple's recognizer, no download), and ask one question per utterance: does this need the assistant? Only the ones that pass would go to a language model. | noul |
| **Clipboard** | What kind of information is on the clipboard, is it what the stated purpose needs, and is it safe to paste as-is. The same decisions are **Shortcuts actions** (“Ask yes/no”, “Classify text”), so a shortcut can branch on meaning without opening the app. | choice + noul + score |
| **Search** | Query × passages: one decision per passage, ranked by its probability — a reranker made of a chat model, no index. | noul or score |

## Run

```bash
# GUI (iPhone or Mac)
xcodegen generate
open Decide.xcodeproj

# headless (macOS), the same function the GUI calls
swift run -c release decide-cli ask --state "The order arrived with the box crushed." \
    --noul "Does the customer want a replacement?" \
    --choice "What is it about?|billing|damaged delivery|how-to" \
    --score "How urgent?|can wait|this week|today"
swift run -c release decide-cli bench --model minicpm5-2b        # shared vs direct prefill
swift run -c release decide-cli --list-models
```

## Measured

Apple M4 Max, macOS 27.0 (26A428), Release, the CLI, sequential engine; the machine was
running other GPU work at the time (timings are an upper bound, not a quiet-machine figure).
A 150-token state and eight questions of 40–60 tokens each; median of 3 runs.

| Model (catalog id) | ms per decision, state shared | ms per decision, every prompt from scratch | prefill + 8 decisions on one state |
|---|---:|---:|---:|
| MiniCPM5 2B int8 (`minicpm5-2b`) | 64.5 | 108.8 | 541 ms vs 1013 ms |
| MiniCPM5 1B int8 (`minicpm5-1b`) | 24.8 | 48.2 | 273 ms vs 468 ms |
| Qwen3 0.6B 4-bit (`qwen3-0.6b`) | 15.8 | 33.6 | 163 ms vs 308 ms |

**Shared** = the state is prefilled once and each question rewinds the engine to it
(`Decision.Timing.reusedTokens` = 150 here); **from scratch** = the whole prompt every time.

Agreement with the published bf16 readout of the same models on the same 144 authored rows
(SemIf's `authored144` fixture, its `direct` rendering byte for byte, its `evaluate.py` metric):

| Model | Prompt tokens identical | Argmax agreement | max / mean \|Δp\| | Mean family balanced accuracy (kit) | Published bf16 |
|---|---:|---:|---:|---:|---:|
| MiniCPM5 2B int8 | 144/144 | 141/144 | 0.183 / 0.020 | 0.681 | 0.686 |
| MiniCPM5 1B int8 | — | — | — | 0.513 | — |
| Qwen3 0.6B 4-bit | 144/144 | 59/144 | 1.000 / 0.580 | 0.333 | 0.440 |

So `minicpm5-2b` is the default: its int8 decisions track the fp32 model. The official 4-bit
Qwen3 0.6B bundle does **not** — it answers the last option far too often — so use it for its
speed only where a wrong decision is cheap. No iPhone numbers yet; the engine path is the same.

## Where the code is

- `Sources/QuickStart.swift` — the take-home: one typed function, no UI. The GUI and the CLI
  both call it.
- `CLI/main.swift` — argument shell over that function, plus `bench` and `oracle` (the two
  tables above).
- `Sources/DecideRuntime.swift` — the one loaded `TypedDecisions` the screens share.
- `Sources/SpeechGate*.swift`, `Clipboard*.swift`, `Search*.swift` — the three screens;
  `Intents.swift` — the Shortcuts actions.

## Integration checklist

- SPM: `github.com/john-rocky/coreai-kit` → product `CoreAIOps` (or `CoreAIKit` for
  `TypedDecisions` alone)
- Info.plist: `NSMicrophoneUsageDescription` and `NSSpeechRecognitionUsageDescription` only
  for the speech gate; the other two screens need nothing
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit` for the 2.7 GB
  MiniCPM5 2B bundle
- First run downloads the model → cached in Application Support (progress callback)
- Measure in Release — Debug is ~3× slower on per-token host work
