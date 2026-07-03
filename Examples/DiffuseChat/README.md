# DiffuseChat

An on-device **diffusion LM** (LLaDA-8B, ~5.3 GB): the reply is not written left-to-right —
a masked canvas denoises in place, tokens popping in in parallel, lowest-entropy first.
`KitDiffusionLM` streams a snapshot per step (still-masked positions as ░) so you watch it
happen.

The whole ML surface is two calls:

```swift
let dlm = try await KitDiffusionLM(catalog: "llada-8b")
let reply = try await dlm.reply(to: "What is the capital of France?") { step in
    print(step.text)   // the live canvas, ░ = still masked
}
```

## Run

```bash
xcodegen generate
open DiffuseChat.xcodeproj
```

Run on a Mac (the shipped bundle is macOS-only today). First launch downloads the model
from the Hugging Face Hub; later launches load from cache.

## CLI (macOS, no Xcode)

```bash
swift run diffuse-cli --prompt "What is the capital of France?"
```

The live canvas streams to stderr; the final reply is stdout (`--quiet` for reply only).
