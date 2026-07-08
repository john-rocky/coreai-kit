import CoreAIKit

// The take-home: load TimesFM 2.5 and forecast a univariate series. This exact function is what
// both the CLI (CLI/main.swift) and the GUI app call — copy it into your own app and it runs.
//
//   let f = try await forecast(series)          // [Float] of any length
//   f.mean        — 128-step point forecast
//   f.quantiles   — 128 × 10 (channel 0 = mean, 1…9 = deciles 0.1…0.9), for fan charts / intervals
public func forecast(_ series: [Float], model: String = "timesfm-2.5-200m") async throws -> Forecast {
    let forecaster = try await KitForecaster(catalog: model)
    return try await forecaster.forecast(series)
}
