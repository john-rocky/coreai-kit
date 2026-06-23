# UpscaleDemo

Minimal CoreAIKit example: **AdcSR ×4 super-resolution** on-device. Pick a photo, tap **Upscale ×4**.

Uses `SuperResolver` (CoreAIKitVision):

```swift
let sr = try await SuperResolver(model: .adcsrX4)   // downloads from the Hub on first use
let big = try await sr.upscale(cgImage)             // ×4, tiled + feather-blended
```

AdcSR (CVPR 2025) is a one-step diffusion-GAN: a pruned Stable Diffusion 2.1 UNet + half VAE decoder
run in a single forward (no prompt, no noise). The graph is one `lr[1,3,128,128] → sr[1,3,512,512]`
tile; `SuperResolver` tiles any-size input. License: Apache-2.0 code + SD-2.1 OpenRAIL++ weights
(commercial-OK). `mlboydaisuke/AdcSR-CoreAI`, fp16 ~870 MB (GPU output matches fp32, cos 1.000008).

## Build

```sh
xcodegen generate
open UpscaleDemo.xcodeproj
```

iOS app (`com.coreaikit.upscaledemo`). Build/run on device.
