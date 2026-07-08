import SwiftUI
import Charts
import CoreAIKit

// Interactive on-device TimesFM 2.5 demo on REAL data. Each series is split into a context (fed to
// the model) and a held-out tail (the true future). The iPhone forecasts the context; we overlay the
// held-out actual so you can see forecast vs. ground truth — all on-device via KitForecaster.

struct RealSeries: Codable, Identifiable {
    var id: String { name }
    let name: String
    let unit: String
    let context: [Float]   // fed to the model
    let actual: [Float]    // held-out true future (overlaid)
}

func loadSeries() -> [RealSeries] {
    guard let url = Bundle.main.url(forResource: "real_series", withExtension: "json"),
          let data = try? Data(contentsOf: url),
          let box = try? JSONDecoder().decode([String: [RealSeries]].self, from: data),
          let s = box["series"] else { return [] }
    return s
}

struct Pt: Identifiable { let id = UUID(); let x: Double; let y: Double }
struct Band: Identifiable { let id = UUID(); let x: Double; let lo: Double; let hi: Double }

@MainActor
final class VM: ObservableObject {
    @Published var series = loadSeries()
    @Published var pick = 0
    @Published var ready = false
    @Published var busy = false
    @Published var status = "loading model…"
    @Published var history: [Pt] = []
    @Published var median: [Pt] = []
    @Published var truth: [Pt] = []
    @Published var band: [Band] = []
    @Published var yDomain: ClosedRange<Double> = 0...1
    @Published var caption = ""

    private var forecaster: KitForecaster?
    static let shown = 120

    func load() async {
        do {
            forecaster = try await KitForecaster(catalog: "timesfm-2.5-200m") { p in
                Task { @MainActor in self.status = "downloading… \(Int(p.fraction * 100))%" }
            }
            ready = true
            await forecast()
        } catch { status = "error: \(error)" }
    }

    func forecast() async {
        guard let f = forecaster, !busy, !series.isEmpty else { return }
        busy = true; status = "forecasting on-device…"
        let s = series[pick]
        do {
            let t0 = Date()
            let out = try await f.forecast(s.context)
            let ms = Int(Date().timeIntervalSince(t0) * 1000)

            let hist = Array(s.context.suffix(Self.shown))
            history = hist.enumerated().map { Pt(x: Double($0.offset), y: Double($0.element)) }
            let base = Double(hist.count - 1)
            let h = s.actual.count                                  // show forecast over the held-out span
            median = (0..<h).map { Pt(x: base + Double($0 + 1), y: Double(out.mean[$0])) }
            band = (0..<h).map {
                Band(x: base + Double($0 + 1), lo: Double(out.quantiles[$0][1]), hi: Double(out.quantiles[$0][9])) }
            truth = s.actual.enumerated().map { Pt(x: base + Double($0.offset + 1), y: Double($0.element)) }

            let mae = zip(out.mean.prefix(h), s.actual).map { abs($0 - $1) }.reduce(0, +) / Float(h)
            let naive = s.actual.map { abs($0 - s.context.last!) }.reduce(0, +) / Float(h)
            caption = String(format: "%@ · %d-step · %d ms · MAE %.2f (%.1f× better than last-value)",
                             s.unit, h, ms, mae, naive / max(mae, 1e-6))

            let ys = hist.map { Double($0) } + out.mean.prefix(h).map { Double($0) } + s.actual.map { Double($0) }
            let lo = ys.min() ?? 0, hi = ys.max() ?? 1, pad = (hi - lo) * 0.08 + 1e-3
            yDomain = (lo - pad)...(hi + pad)
            status = s.name
        } catch { status = "error: \(error)" }
        busy = false
    }
}

struct ContentView: View {
    @StateObject private var vm = VM()
    private let nowX = Double(VM.shown - 1)

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("TimesFM 2.5 · on-device forecast vs. actual").font(.headline)
            Text(vm.status).font(.subheadline).foregroundStyle(.secondary)

            Picker("series", selection: $vm.pick) {
                ForEach(Array(vm.series.enumerated()), id: \.offset) { i, s in Text(s.name).tag(i) }
            }
            .pickerStyle(.segmented)
            .disabled(!vm.ready || vm.busy)
            .onChange(of: vm.pick) { Task { await vm.forecast() } }

            chart.frame(maxHeight: .infinity)

            Text(vm.caption).font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 16) {
                Label("history", systemImage: "minus").foregroundStyle(.secondary)
                Label("forecast", systemImage: "minus").foregroundStyle(.orange)
                Label("actual", systemImage: "minus").foregroundStyle(.teal)
                Label("p10–p90", systemImage: "rectangle.fill").foregroundStyle(.orange.opacity(0.4))
            }.font(.caption2)
        }
        .padding()
        .task { await vm.load() }
    }

    private var chart: some View {
        Chart {
            ForEach(vm.band) {
                AreaMark(x: .value("t", $0.x), yStart: .value("p10", $0.lo), yEnd: .value("p90", $0.hi))
                    .foregroundStyle(.orange.opacity(0.16))
            }
            ForEach(vm.history) {
                LineMark(x: .value("t", $0.x), y: .value("v", $0.y), series: .value("s", "history"))
                    .foregroundStyle(.secondary)
            }
            ForEach(vm.truth) {
                LineMark(x: .value("t", $0.x), y: .value("v", $0.y), series: .value("s", "actual"))
                    .foregroundStyle(.teal)
            }
            ForEach(vm.median) {
                LineMark(x: .value("t", $0.x), y: .value("v", $0.y), series: .value("s", "forecast"))
                    .foregroundStyle(.orange).lineStyle(StrokeStyle(lineWidth: 2))
            }
            if !vm.median.isEmpty {
                RuleMark(x: .value("now", nowX))
                    .foregroundStyle(.gray.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }
        }
        .chartYScale(domain: vm.yDomain)
        .chartYAxis { AxisMarks(position: .leading) }
        .animation(.easeInOut(duration: 0.35), value: vm.median.map(\.y))
    }
}

@main
struct ForecastApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}
