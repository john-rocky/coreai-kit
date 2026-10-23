# Forecast — time-series forecasting on device

Give it your own univariate series — paste numbers, pick a CSV file, or use one of the bundled real series — and get
a probabilistic forecast back, fully on device. Two models:

| Model | Horizon | Output | Catalog id |
|---|---|---|---|
| [TimesFM 2.5 200M](https://john-rocky.github.io/coreai-model-zoo/models/timesfm/) | 128 steps | mean + deciles 0.1…0.9 | `timesfm-2.5-200m` |
| [Timestral 209M](https://huggingface.co/mistralai/Timestral-209M) | any, 1…1,024 steps | 21 quantiles 0.01…0.99 | `timestral-209m` |

The whole ML surface is two lines per model:

```swift
let forecaster = try await KitForecaster(catalog: "timesfm-2.5-200m")
let f = try await forecaster.forecast(series)                 // series: [Float]; f.mean, f.quantiles (128 × 10)

let timestral = try await KitTimestralForecaster(catalog: "timestral-209m")
let q = try await timestral.forecast(series, horizon: 96)     // NaN in series = a missing value
q.median, q.quantile(0.1), q.quantile(0.9)                    // 96 values each
```

`Sources/QuickStart.swift` holds exactly those functions (`forecast`, `forecastQuantiles`, `loadTimestral`), and both
the app and the CLI call them; `Sources/SeriesInput.swift` turns pasted text or a CSV file into a series. Copy the two
files into your own project and they run.

## Your own series

- **Paste**: one number per line, or numbers separated by commas, semicolons, tabs or spaces.
- **CSV**: a table with a header row or a date column — the last column is the series (`date,units` works as is).
- **Missing values**: an empty cell, `NaN`, `NA`, `N/A`, `null`, `-` or `?`. The models skip them.
- The decimal mark is `.`; a value that is not a number stops the forecast with its line: `Line 3: 'abc' is not a number.`

Timestral reads at most its bucket's context (2,048 points for the 64-patch graph, 8,192 for the 256-patch one); a
longer series keeps its most recent points, and the forecast says how many it read. A series with no number at all,
an empty input, or a horizon outside 1…1,024 is refused with the reason.

## Run

```bash
swift run forecast-cli --help                                              # terminal
swift run forecast-cli --model timestral-209m --series sales.csv --horizon 48 --band 90
pbpaste | swift run forecast-cli --model timestral-209m --series - --horizon 24
# --compute auto (default: each graph on its form's unit) | gpu | ane (that unit for every graph)

xcodegen generate                                                          # app
open Forecast.xcodeproj
```

Each model downloads its bundle from the Hugging Face Hub once and caches it; later runs are local. The CLI also runs a
local Timestral bundle — a graph (`.aimodel` / `.aimodelc`) or a folder of them — with `--bundle`.

## Notes

- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit` (`Forecast.entitlements`).
  TimesFM 2.5 needs it on an iPhone 18 Pro: without it, the app aborted while TimesFM's graph was compiled at load.
- TimesFM takes a fixed 2048-point context; `KitForecaster` front-pads and masks shorter series. Timestral runs one
  graph per context bucket and picks the smallest bucket that holds the series; the bucket, its rows per call and its
  input type are read from the bundle itself, so any exported form of the graph runs through the same code.
- Each Timestral graph runs on the compute unit of its form, which the kit reads from the graph's name: the Neural
  Engine form (`…_fp16p7-qkv4-tanh-conv`) on the Neural Engine, the GPU forms (`…_fp16-attn32`, `…_fp32`) on the GPU —
  leave `computeUnits` out and a mixed folder (the Mac's short buckets are the Neural Engine form, its 8,192-step one a
  GPU form) runs each graph where it computes right. The Neural Engine form computes in fp16 throughout: on the GPU its
  attention scores saturate at ±65,504 and the forecast would come out finite and wrong, so `computeUnits: .gpu` (or
  `--compute gpu`) for it stops at load with the reason, and a forecast whose graph output reaches fp16's limit stops
  with an error instead of being returned.
- Both are one stateless graph plus host arithmetic — no KV cache, no state to carry between calls. Timestral's host
  (`TimestralHost`) is a port of the released forecaster's defaults: causal float64 scaling, symmetric averaging
  (the series and its negation), the refine frame, the cap, and clipping at 0 for a series with no negative value. It
  is gated window by window against the Python reference host on the model zoo's fixture.
- Timestral v1 is univariate and one decode block (1,024 steps); covariates and longer horizons are not ported.

## Licence of the Timestral weights

Timestral-209M is released by Mistral AI under a **Modified MIT License** (attribution notice: 2026 - Mistral AI).
Two conditions matter for an app:

1. The attribution notice and the permission notice go with every copy of the model or a substantial portion of it.
2. **You may not exercise any rights under the licence if the global consolidated monthly revenue of your company (or
   of your employer) exceeded $20 million (or its equivalent in another currency) in the preceding month.** This
   applies to the model and to any derivative, modification or combined work — the converted Core AI graph included.
   Mistral AI offers commercial licences (sales@mistral.ai).

Read the full terms before shipping:
[LICENSE](https://huggingface.co/mistralai/Timestral-209M/blob/9874d252ab0aae16f7f7c999f98f98d6ad7d8365/LICENSE).
Once listed, its catalog entry carries `license`, so an app can show it.
