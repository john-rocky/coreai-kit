import CoreAIKit
import Foundation

// The take-home: load a forecasting model and forecast a univariate series. These exact functions are what both the
// CLI (CLI/main.swift) and the GUI app call — copy this file into your own app and it runs.
//
//   let f = try await forecast(series)                          // TimesFM 2.5, [Float] of any length
//   f.mean        — 128-step point forecast
//   f.quantiles   — 128 × 10 (channel 0 = mean, 1…9 = deciles 0.1…0.9), for fan charts / intervals
//
//   let q = try await forecastQuantiles(series, horizon: 96)    // Timestral 209M, up to 1,024 steps
//   q.median, q.quantile(0.1), q.quantile(0.9)                  // 96 values each; 21 levels 0.01…0.99
public func forecast(_ series: [Float], model: String = "timesfm-2.5-200m") async throws -> Forecast {
    let forecaster = try await KitForecaster(catalog: model)
    return try await forecaster.forecast(series)
}

/// One Timestral 209M forecast. NaN in `series` is a missing value.
@available(macOS 27, iOS 27, *)
public func forecastQuantiles(_ series: [Float], horizon: Int, bundle: URL? = nil) async throws -> QuantileForecast {
    try await loadTimestral(bundle: bundle).forecast(series, horizon: horizon)
}

/// Timestral 209M from the catalog, or from a local bundle (one graph, or a folder of graphs) when `bundle` is given.
/// Each graph runs on the compute unit of its form (the Neural Engine form on the Neural Engine, the others on the
/// GPU). Keep it for the next forecasts: loading is the slow part.
@available(macOS 27, iOS 27, *)
public func loadTimestral(bundle: URL? = nil) async throws -> KitTimestralForecaster {
    if let bundle { return try await KitTimestralForecaster(bundleAt: bundle) }
    return try await KitTimestralForecaster()
}
