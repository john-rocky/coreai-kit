# Forecast — time-series forecasting on device

Give it a univariate series of any length and get a 128-step forecast back, fully on device.
The model is [TimesFM 2.5](https://john-rocky.github.io/coreai-model-zoo/models/timesfm/), a
decoder-only forecasting foundation model — about 25 ms per forecast on an iPhone 17 Pro.

The whole ML surface is two lines:

```swift
let forecaster = try await KitForecaster(catalog: "timesfm-2.5-200m")
let f = try await forecaster.forecast(series)   // series: [Float]
```

`f.mean` is the 128-step point forecast. `f.quantiles` is 128 × 10 — channel 0 is the mean,
1…9 are the deciles 0.1…0.9 — which is what you draw a fan chart or a prediction interval from.
`Sources/QuickStart.swift` holds exactly that function, and both the app and the CLI call it;
copy the file into your own project and it runs.

## Run

```bash
swift run forecast-cli --help          # terminal

xcodegen generate                      # app
open Forecast.xcodeproj
```

First use downloads the bundle (~463 MB) from the Hugging Face Hub and caches it; later runs
are local.

## Notes

- The model takes a fixed 2048-point context. `KitForecaster` front-pads and masks shorter
  series for you, so a series of any length is valid input.
- One stateless graph plus host-side RevIN normalisation and flipping — no KV cache, no state
  to carry between calls, so each forecast is independent.
