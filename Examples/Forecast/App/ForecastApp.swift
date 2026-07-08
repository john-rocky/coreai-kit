import SwiftUI
import Charts
import CoreAIKit

// Interactive on-device TimesFM 2.5 demo. Pick a pattern (or shuffle a noisy one); the app forecasts
// 128 steps on the iPhone GPU and draws a fan chart (history + median + p10–p90 band). Everything
// runs on-device via KitForecaster — no server.

struct Sample: Identifiable, Sendable {
    let id = UUID(); let name: String; let make: @Sendable (Int) -> [Float]
}

enum Samples {
    static let all: [Sample] = [
        Sample(name: "Seasonal") { seed in
            (0..<512).map { 10 + 3 * sinf(2 * .pi * Float($0) / 48) + 0.01 * Float($0)
                + 0.4 * n($0, seed) } },
        Sample(name: "Trend") { seed in
            (0..<512).map { 5 + 0.03 * Float($0) + 2 * sinf(2 * .pi * Float($0) / 24) + 0.5 * n($0, seed) } },
        Sample(name: "Daily+Weekly") { seed in
            (0..<512).map { 12 + 3 * sinf(2 * .pi * Float($0) / 24) + 1.6 * sinf(2 * .pi * Float($0) / 168)
                + 0.5 * n($0, seed) } },
        Sample(name: "Damped") { seed in
            (0..<512).map { 18 + 9 * expf(-Float($0) / 260) * sinf(2 * .pi * Float($0) / 30) + 0.4 * n($0, seed) } },
        Sample(name: "Walk") { seed in
            var v: Float = 40; return (0..<512).map { i -> Float in v += n(i, seed) + 0.02 * sinf(Float(i) / 12); return v } },
    ]
    // deterministic pseudo-noise (no RNG => reproducible)
    static func n(_ i: Int, _ seed: Int) -> Float {
        let x = sinf(Float(i) * 12.9898 + Float(seed) * 7.233) * 43758.547
        return (x - x.rounded(.down)) * 2 - 1
    }
}

struct Pt: Identifiable { let id = UUID(); let x: Double; let y: Double }
struct Band: Identifiable { let id = UUID(); let x: Double; let lo: Double; let hi: Double }

@MainActor
final class VM: ObservableObject {
    @Published var ready = false
    @Published var busy = false
    @Published var status = "loading model…"
    @Published var pick = 0
    @Published var seed = 1
    @Published var history: [Pt] = []
    @Published var median: [Pt] = []
    @Published var band: [Band] = []
    @Published var yDomain: ClosedRange<Double> = 0...20
    @Published var lastMs = 0

    private var forecaster: KitForecaster?
    static let shown = 96

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
        guard let f = forecaster, !busy else { return }
        busy = true; status = "forecasting on-device…"
        let series = Samples.all[pick].make(seed)
        do {
            let t0 = Date()
            let out = try await f.forecast(series)
            lastMs = Int(Date().timeIntervalSince(t0) * 1000)

            let hist = Array(series.suffix(Self.shown))
            history = hist.enumerated().map { Pt(x: Double($0.offset), y: Double($0.element)) }
            let base = Double(Self.shown - 1)
            median = out.mean.enumerated().map { Pt(x: base + Double($0.offset + 1), y: Double($0.element)) }
            band = (0..<out.mean.count).map {
                Band(x: base + Double($0 + 1), lo: Double(out.quantiles[$0][1]), hi: Double(out.quantiles[$0][9])) }
            let ys = hist.map { Double($0) } + out.mean.map { Double($0) }
            yDomain = ((ys.min() ?? 0) - 1).rounded(.down)...((ys.max() ?? 20) + 1).rounded(.up)
            status = "\(Samples.all[pick].name) · \(lastMs) ms on-device"
        } catch { status = "error: \(error)" }
        busy = false
    }
}

struct ContentView: View {
    @StateObject private var vm = VM()
    private let nowX = Double(VM.shown - 1)

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("TimesFM 2.5 · on-device forecasting").font(.headline)
            Text(vm.status).font(.caption).foregroundStyle(.secondary)

            Picker("pattern", selection: $vm.pick) {
                ForEach(Array(Samples.all.enumerated()), id: \.offset) { i, s in Text(s.name).tag(i) }
            }
            .pickerStyle(.segmented)
            .disabled(!vm.ready || vm.busy)
            .onChange(of: vm.pick) { Task { await vm.forecast() } }

            chart.frame(maxHeight: .infinity)

            HStack {
                HStack(spacing: 14) {
                    Label("history", systemImage: "minus").foregroundStyle(.secondary)
                    Label("forecast", systemImage: "minus").foregroundStyle(.orange)
                    Label("p10–p90", systemImage: "rectangle.fill").foregroundStyle(.orange.opacity(0.4))
                }.font(.caption2)
                Spacer()
                Button { vm.seed += 1; Task { await vm.forecast() } } label: {
                    Label("shuffle", systemImage: "shuffle")
                }.font(.caption).disabled(!vm.ready || vm.busy)
            }
        }
        .padding()
        .task { await vm.load() }
    }

    private var chart: some View {
        Chart {
            ForEach(vm.band) {
                AreaMark(x: .value("t", $0.x), yStart: .value("p10", $0.lo), yEnd: .value("p90", $0.hi))
                    .foregroundStyle(.orange.opacity(0.18))
            }
            ForEach(vm.history) {
                LineMark(x: .value("t", $0.x), y: .value("v", $0.y), series: .value("s", "history"))
                    .foregroundStyle(.secondary)
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
