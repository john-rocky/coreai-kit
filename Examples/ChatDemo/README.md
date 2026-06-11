# ChatDemo

Minimal multiplatform chat app on CoreAIKit: pick a starter model, it downloads from the
Hugging Face Hub on first use, then chat with live stats (load / TTFT / tok/s / memory).

## Run

```bash
xcodegen generate
open ChatDemo.xcodeproj
```

Run the `ChatDemo` scheme on a macOS or iPhone destination (real device — the CoreAI
framework is not in the iOS Simulator SDK). The scheme runs Release — the engine's
per-token host work is ~3x slower in Debug.

Downloaded models cache under `Application Support/CoreAIKit/Models`.
On iOS, only models with a published `ios/` variant appear in the picker.
