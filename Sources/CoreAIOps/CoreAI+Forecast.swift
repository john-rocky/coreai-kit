// CoreAI+Forecast.swift — anchored time-series forecasting over local catalog models.
//
// ```swift
// let forecast = try await CoreAI.forecast(dailySales)   // 128 steps ahead, ~25 ms warm
// print(forecast.mean.prefix(7))                         // next week, point forecast
// ```

import CoreAIKit
import Foundation

extension CoreAI {
    /// Default forecasting model: TimesFM 2.5 200M.
    public static let defaultForecastModel = "timesfm-2.5-200m"

    /// Number series → 128-step forecast (univariate, any length ≥ 1; the last 2048
    /// points are used). `Forecast.mean` is the point forecast; `quantiles` carry the
    /// uncertainty bands. Slice `mean.prefix(h)` for a shorter horizon.
    public static func forecast(
        _ series: [Float], options: OpOptions = OpOptions()
    ) async throws -> Forecast {
        let forecaster = try await ForecastOpModels.shared.forecaster(
            catalog: options.model ?? defaultForecastModel)
        return try await forecaster.forecast(series)
    }
}

/// Process-wide cache of loaded forecasters, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached.
actor ForecastOpModels {
    static let shared = ForecastOpModels()

    private var forecasterLoads: [String: Task<KitForecaster, Error>] = [:]

    func forecaster(catalog id: String) async throws -> KitForecaster {
        if let load = forecasterLoads[id] { return try await load.value }
        let load = Task<KitForecaster, Error> {
            try await KitForecaster(catalog: id, downloadProgress: OpDownloads.forward)
        }
        forecasterLoads[id] = load
        do {
            return try await load.value
        } catch {
            forecasterLoads[id] = nil
            throw error
        }
    }
}
