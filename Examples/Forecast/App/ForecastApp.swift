import Charts
import CoreAIKit
import SwiftUI
import UniformTypeIdentifiers

// On-device forecasting of your own series. Pick a model (TimesFM 2.5, or Timestral 209M for any horizon up to
// 1,024 steps with 21 quantiles), give a series — a bundled real one (its held-out tail is drawn as the actual), pasted
// numbers, or a CSV file — choose the horizon, and the forecast comes back as a fan chart: the median and the band.

struct RealSeries: Codable, Identifiable {
    var id: String { name }
    let name: String
    let unit: String
    let context: [Float]  // fed to the model
    let actual: [Float]  // held-out true future (overlaid)
}

func loadSeries() -> [RealSeries] {
    guard let url = Bundle.main.url(forResource: "real_series", withExtension: "json"),
        let data = try? Data(contentsOf: url),
        let box = try? JSONDecoder().decode([String: [RealSeries]].self, from: data),
        let s = box["series"]
    else { return [] }
    return s
}

struct Pt: Identifiable {
    let id = UUID()
    let x: Double
    let y: Double
}

struct BandPt: Identifiable {
    let id = UUID()
    let x: Double
    let lo: Double
    let hi: Double
}

enum ForecastModel: String, CaseIterable, Identifiable {
    case timestral = "Timestral 209M"
    case timesfm = "TimesFM 2.5"
    var id: String { rawValue }
}

enum SeriesSource: String, CaseIterable, Identifiable {
    case sample = "Sample"
    case paste = "Paste"
    case csv = "CSV file"
    var id: String { rawValue }
}

/// The interval drawn around the median: its lower and upper quantile levels.
enum Band: String, CaseIterable, Identifiable {
    case p80 = "80%"
    case p90 = "90%"
    case p98 = "98%"
    var id: String { rawValue }
    var levels: (Float, Float) {
        switch self {
        case .p80: return (0.1, 0.9)
        case .p90: return (0.05, 0.95)
        case .p98: return (0.01, 0.99)
        }
    }
}

enum ForecastAppError: LocalizedError {
    case timestralBundleMissing
    case timesfmHorizon(Int)
    case horizonNotANumber(String)

    var errorDescription: String? {
        switch self {
        case .timestralBundleMissing:
            return "Timestral 209M is not in this kit's catalog yet. Put its bundle in the app's Documents/models/timestral/ "
                + "folder (Files app → On My iPhone → Forecast)."
        case .timesfmHorizon(let h):
            return "TimesFM 2.5 forecasts 1…128 steps; \(h) is outside. Timestral 209M goes to 1,024."
        case .horizonNotANumber(let s):
            return "Horizon '\(s)' is not a whole number of steps."
        }
    }
}

@MainActor
final class ForecastViewModel: ObservableObject {
    @Published var model: ForecastModel = .timestral
    @Published var source: SeriesSource = .sample
    @Published var samples = loadSeries()
    @Published var pick = 0
    @Published var pasted = ""
    @Published var csvName: String?
    @Published var horizonText = "48"
    @Published var band: Band = .p80
    @Published var busy = false
    @Published var status = "Pick a series and forecast."
    @Published var error: String?
    @Published var history: [Pt] = []
    @Published var median: [Pt] = []
    @Published var truth: [Pt] = []
    @Published var bandPoints: [BandPt] = []
    @Published var yDomain: ClosedRange<Double> = 0...1
    @Published var nowX = 0.0
    @Published var caption = ""

    var csvText: String?
    /// A scripted run's `-compute gpu|ane`: that unit for every Timestral graph. nil = each graph on its form's unit.
    var computeOverride: String?
    /// A local Timestral bundle to load instead of the catalog entry (a graph, or a folder of graphs).
    var localBundle: URL?
    /// Called after every forecast or error (the scripted runs log it).
    var onResult: ((_ outcome: [String: Any]) -> Void)?
    private var timesfm: KitForecaster?
    private var timestral: KitTimestralForecaster?
    private(set) var lastLoadMilliseconds: Double?

    /// The series the current input gives, NaN where a value is missing.
    func currentSeries() throws -> (values: [Float], actual: [Float]?, label: String) {
        switch source {
        case .sample:
            guard samples.indices.contains(pick) else { throw SeriesInputError.empty }
            let s = samples[pick]
            return (s.context, s.actual, s.name)
        case .paste:
            let input = try SeriesParser.parse(pasted)
            return (input.values, nil, "pasted series")
        case .csv:
            guard let text = csvText else { throw SeriesInputError.empty }
            let input = try SeriesParser.parse(text)
            return (input.values, nil, csvName.map { name in input.column.map { "\(name) · \($0)" } ?? name } ?? "CSV")
        }
    }

    func importCSV(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            csvText = try String(contentsOf: url, encoding: .utf8)
            csvName = url.lastPathComponent
            error = nil
        } catch {
            self.error = "Cannot read \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }

    /// The loaded Timestral forecaster (loads it on first use), for the scripted fixture replay.
    func timestralForecaster() async throws -> KitTimestralForecaster { try await loadTimestral() }

    private func loadTimestral() async throws -> KitTimestralForecaster {
        if let t = timestral { return t }
        let t0 = Date()
        let loaded: KitTimestralForecaster
        if let localBundle {
            loaded = try await openTimestral(at: localBundle)
            lastLoadMilliseconds = Date().timeIntervalSince(t0) * 1000
            timestral = loaded
            return loaded
        }
        do {
            let progress: @Sendable (DownloadProgress) -> Void = { p in
                Task { @MainActor in self.status = "Downloading Timestral 209M… \(Int(p.fraction * 100))%" }
            }
            if let unit = computeOverride {
                loaded = try await KitTimestralForecaster(
                    computeUnits: unit == "ane" ? .neuralEngine : .gpu, downloadProgress: progress)
            } else {
                loaded = try await KitTimestralForecaster(downloadProgress: progress)
            }
        } catch CoreAIKitError.modelNotInCatalog {
            // Until the catalog lists it: a bundle the user put in Documents/models/timestral/.
            let local = URL.documentsDirectory.appending(path: "models/timestral")
            guard FileManager.default.fileExists(atPath: local.path) else { throw ForecastAppError.timestralBundleMissing }
            loaded = try await openTimestral(at: local)
        }
        lastLoadMilliseconds = Date().timeIntervalSince(t0) * 1000
        timestral = loaded
        return loaded
    }

    /// A local bundle: each graph on its form's unit, or the scripted run's `-compute` unit for every graph.
    private func openTimestral(at url: URL) async throws -> KitTimestralForecaster {
        guard let unit = computeOverride else { return try await KitTimestralForecaster(bundleAt: url) }
        return try await KitTimestralForecaster(bundleAt: url, computeUnits: unit == "ane" ? .neuralEngine : .gpu)
    }

    /// Where a forecast's graph ran, for the caption and the scripted log.
    static func unitName(_ d: QuantileForecast.Details) -> String {
        switch d.computeUnits {
        case .neuralEngine: return "Neural Engine"
        case .gpu: return "GPU"
        default: return "CPU"
        }
    }

    private func loadTimesFM() async throws -> KitForecaster {
        if let f = timesfm { return f }
        let t0 = Date()
        let loaded = try await KitForecaster(catalog: "timesfm-2.5-200m") { p in
            Task { @MainActor in self.status = "Downloading TimesFM 2.5… \(Int(p.fraction * 100))%" }
        }
        lastLoadMilliseconds = Date().timeIntervalSince(t0) * 1000
        timesfm = loaded
        return loaded
    }

    func forecast() async {
        guard !busy else { return }
        busy = true
        error = nil
        defer { busy = false }
        var outcome: [String: Any] = ["model": model.rawValue, "source": source.rawValue, "horizon_text": horizonText]
        do {
            let text = horizonText.trimmingCharacters(in: .whitespaces)
            guard let h = Int(text) else { throw ForecastAppError.horizonNotANumber(text) }
            let (series, actual, label) = try currentSeries()
            outcome["values"] = series.count
            let lo: [Float], mid: [Float], hi: [Float]
            let detail: String
            let t0 = Date()
            switch model {
            case .timestral:
                status = "Loading Timestral 209M…"
                let f = try await loadTimestral()
                status = "Forecasting on device…"
                let q = try await f.forecast(series, horizon: h)
                let (l, u) = band.levels
                lo = q.quantile(l)!
                mid = q.median
                hi = q.quantile(u)!
                let d = q.details
                let dropped = d.contextDropped > 0 ? " (oldest \(d.contextDropped.formatted()) left out)" : ""
                detail = String(format: "%.1f ms (graph %.1f ms)", d.totalMilliseconds, d.graphMilliseconds)
                    + " on the \(Self.unitName(d)) · read \(d.contextUsed.formatted()) points\(dropped)"
                outcome.merge([
                    "total_ms": d.totalMilliseconds, "graph_ms": d.graphMilliseconds, "bucket": "\(d.bucket)",
                    "compute": Self.unitName(d), "context_used": d.contextUsed, "context_dropped": d.contextDropped,
                ]) { $1 }
            case .timesfm:
                guard (1...128).contains(h) else { throw ForecastAppError.timesfmHorizon(h) }
                status = "Loading TimesFM 2.5…"
                let f = try await loadTimesFM()
                status = "Forecasting on device…"
                let ts = Date()
                let out = try await f.forecast(series)
                let ms = Date().timeIntervalSince(ts) * 1000
                lo = (0..<h).map { out.quantiles[$0][1] }
                mid = Array(out.mean.prefix(h))
                hi = (0..<h).map { out.quantiles[$0][9] }
                detail = String(format: "%.1f ms", ms) + " · p10–p90"
                outcome["total_ms"] = ms
            }
            let wall = Date().timeIntervalSince(t0) * 1000
            draw(series: series, lo: lo, mid: mid, hi: hi, actual: actual)
            var line = "\(model.rawValue) · \(h)-step forecast · \(detail)"
            if let actual, !actual.isEmpty {
                let n = min(h, actual.count)
                let mae = zip(mid.prefix(n), actual.prefix(n)).map { abs($0 - $1) }.reduce(0, +) / Float(n)
                let last = series.last(where: { $0.isFinite }) ?? 0
                let naive = actual.prefix(n).map { abs($0 - last) }.reduce(0, +) / Float(n)
                line += String(format: " · MAE %.2f (last value %.2f)", mae, naive)
            }
            caption = line
            status = label
            outcome.merge(["ok": true, "wall_ms": wall, "load_ms": lastLoadMilliseconds.map { $0 as Any } ?? NSNull()]) { $1 }
        } catch {
            self.error = error.localizedDescription
            status = "No forecast."
            outcome.merge(["ok": false, "error": error.localizedDescription]) { $1 }
        }
        onResult?(outcome)
    }

    private func draw(series: [Float], lo: [Float], mid: [Float], hi: [Float], actual: [Float]?) {
        let h = mid.count
        let shown = min(series.count, max(96, 2 * h), 1024)
        let tail = Array(series.suffix(shown))
        history = tail.enumerated().compactMap { $0.element.isFinite ? Pt(x: Double($0.offset), y: Double($0.element)) : nil }
        let base = Double(tail.count - 1)
        nowX = base
        // The median, the band (at zero width) and the actual start at the last observed point, so each one
        // continues the history line instead of starting a step to its right.
        let last = history.last
        func fromLast(_ ys: [Float]) -> [Pt] {
            let start = last.map { [Pt(x: $0.x, y: $0.y)] } ?? []
            return start + ys.enumerated().map { Pt(x: base + Double($0.offset + 1), y: Double($0.element)) }
        }
        median = fromLast(mid)
        let bandStart = last.map { [BandPt(x: $0.x, lo: $0.y, hi: $0.y)] } ?? []
        bandPoints = bandStart + (0..<h).map { BandPt(x: base + Double($0 + 1), lo: Double(lo[$0]), hi: Double(hi[$0])) }
        let truthTail = Array((actual ?? []).prefix(h))
        truth = truthTail.isEmpty ? [] : fromLast(truthTail)
        let ys = history.map(\.y) + bandPoints.flatMap { [$0.lo, $0.hi] } + truth.map(\.y)
        let low = ys.min() ?? 0, high = ys.max() ?? 1
        let pad = (high - low) * 0.08 + 1e-3
        yDomain = (low - pad)...(high + pad)
    }
}

struct ContentView: View {
    @ObservedObject var vm: ForecastViewModel
    @State private var importing = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Picker("Model", selection: $vm.model) {
                        ForEach(ForecastModel.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)

                    Picker("Series", selection: $vm.source) {
                        ForEach(SeriesSource.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)

                    input

                    HStack(spacing: 10) {
                        Text("Horizon")
                        TextField("steps", text: $vm.horizonText)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 80)
                        Text("steps").foregroundStyle(.secondary)
                        Spacer()
                        Picker("Band", selection: $vm.band) {
                            ForEach(Band.allCases) { Text($0.rawValue).tag($0) }
                        }
                        .pickerStyle(.menu)
                        .disabled(vm.model == .timesfm)
                    }

                    Button {
                        Task { await vm.forecast() }
                    } label: {
                        Text(vm.busy ? "Forecasting…" : "Forecast").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.busy)

                    if let error = vm.error {
                        Text(error).font(.callout).foregroundStyle(.red)
                    }
                    Text(vm.status).font(.subheadline).foregroundStyle(.secondary)
                    chart.frame(height: 260)
                    Text(vm.caption).font(.caption).foregroundStyle(.secondary)
                    legend
                }
                .padding()
            }
            .navigationTitle("Forecast")
            .navigationBarTitleDisplayMode(.inline)
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.commaSeparatedText, .tabSeparatedText, .plainText]) {
            if case .success(let url) = $0 { vm.importCSV(url) }
        }
    }

    @ViewBuilder private var input: some View {
        switch vm.source {
        case .sample:
            Picker("Sample", selection: $vm.pick) {
                ForEach(Array(vm.samples.enumerated()), id: \.offset) { i, s in Text("\(s.name) · \(s.unit)").tag(i) }
            }
            .pickerStyle(.menu)
        case .paste:
            VStack(alignment: .leading, spacing: 6) {
                TextEditor(text: $vm.pasted)
                    .font(.system(.footnote, design: .monospaced))
                    .frame(height: 110)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                HStack {
                    PasteButton(payloadType: String.self) { strings in
                        if let s = strings.first { vm.pasted = s }
                    }
                    .labelStyle(.titleAndIcon)
                    Text("one number per line, or separated by commas").font(.caption).foregroundStyle(.secondary)
                }
            }
        case .csv:
            HStack {
                Button("Choose CSV…") { importing = true }
                    .buttonStyle(.bordered)
                Text(vm.csvName ?? "no file").font(.callout).foregroundStyle(.secondary).lineLimit(1)
            }
        }
    }

    private var chart: some View {
        Chart {
            ForEach(vm.bandPoints) {
                AreaMark(x: .value("t", $0.x), yStart: .value("lo", $0.lo), yEnd: .value("hi", $0.hi))
                    .foregroundStyle(.orange.opacity(0.22))
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
                LineMark(x: .value("t", $0.x), y: .value("v", $0.y), series: .value("s", "median"))
                    .foregroundStyle(.orange)
                    .lineStyle(StrokeStyle(lineWidth: 2))
            }
            if !vm.median.isEmpty {
                RuleMark(x: .value("now", vm.nowX))
                    .foregroundStyle(.gray.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }
        }
        .chartYScale(domain: vm.yDomain)
        .chartYAxis { AxisMarks(position: .leading) }
    }

    private var legend: some View {
        HStack(spacing: 14) {
            Label("history", systemImage: "minus").foregroundStyle(.secondary)
            Label("median", systemImage: "minus").foregroundStyle(.orange)
            if !vm.truth.isEmpty { Label("actual", systemImage: "minus").foregroundStyle(.teal) }
            Label(vm.model == .timesfm ? "p10–p90" : vm.band.rawValue + " band", systemImage: "rectangle.fill")
                .foregroundStyle(.orange.opacity(0.5))
        }
        .font(.caption2)
    }
}

@main
struct ForecastApp: App {
    @StateObject private var vm = ForecastViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(vm: vm)
                .scriptedRun(vm)
        }
    }
}
