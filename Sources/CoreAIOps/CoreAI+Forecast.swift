// CoreAI+Forecast.swift — anchored time-series forecasting over local catalog models.
//
// ```swift
// let forecast = try await CoreAI.forecast(dailySales)   // 128 steps ahead, ~25 ms warm
// print(forecast.mean.prefix(7))                         // next week, point forecast
//
// let q = try await CoreAI.forecastQuantiles(dailySales, horizon: 30)   // Timestral 209M, 21 quantiles a step
// print(q.median.prefix(7), q.quantile(0.9)!.prefix(7))
// ```

import CoreAIKit
import Foundation

extension CoreAI {
    /// Default forecasting model: TimesFM 2.5 200M.
    public static let defaultForecastModel = "timesfm-2.5-200m"

    /// Number series → 128-step forecast (univariate, any length ≥ 1; the last 2048
    /// points are used). `Forecast.mean` is the point forecast; `quantiles` carry the
    /// uncertainty bands. Slice `mean.prefix(h)` for a shorter horizon.
    @available(macOS 27, iOS 27, *)
    public static func forecast(
        _ series: [Float], options: OpOptions = OpOptions()
    ) async throws -> Forecast {
        let id = options.model ?? defaultForecastModel
        let forecaster = try await ForecastOpModels.shared.forecaster(catalog: id)
        return try await withPinnedModel(ResidentKind.forecaster, id) {
            try await forecaster.forecast(series)
        }
    }
}

extension CoreAI {
    /// Default model of `forecastQuantiles`: Timestral 209M.
    public static let defaultQuantileForecastModel = "timestral-209m"

    /// Number series → `horizon` steps of 21 quantiles each (0.01 … 0.99; univariate, any length ≥ 1, NaN = a
    /// missing value; at most 1,024 steps). A series longer than the model reads keeps its most recent points.
    ///
    /// ```swift
    /// let f = try await CoreAI.forecastQuantiles(hourlyLoad, horizon: 48)
    /// print(f.median.prefix(6), f.quantile(0.1)!.prefix(6), f.quantile(0.9)!.prefix(6))
    /// ```
    @available(macOS 27, iOS 27, *)
    public static func forecastQuantiles(
        _ series: [Float], horizon: Int, options: OpOptions = OpOptions()
    ) async throws -> QuantileForecast {
        let id = options.model ?? defaultQuantileForecastModel
        let forecaster = try await ForecastOpModels.shared.quantileForecaster(catalog: id)
        return try await withPinnedModel(ResidentKind.forecaster, id) {
            try await forecaster.forecast(series, horizon: horizon)
        }
    }
}

/// Process-wide cache of loaded forecasters, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached.
@available(macOS 27, iOS 27, *)
actor ForecastOpModels {
    static let shared = ForecastOpModels()

    private let forecasters = ResidentCache<KitForecaster>(kind: ResidentKind.forecaster)
    private let quantileForecasters = ResidentCache<KitTimestralForecaster>(kind: ResidentKind.forecaster)

    func forecaster(catalog id: String) async throws -> KitForecaster {
        try await forecasters.value(for: id) {
            try await KitForecaster(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }

    func quantileForecaster(catalog id: String) async throws -> KitTimestralForecaster {
        try await quantileForecasters.value(for: id) {
            try await KitTimestralForecaster(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }
}
