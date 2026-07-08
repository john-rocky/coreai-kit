import SwiftUI
import CoreAIKit

// Minimal on-device self-test for TimesFM 2.5. On launch it downloads the model (AOT .aimodelc on
// iOS), forecasts a fixed demo series through KitForecaster, and prints a `GATE:` line to the
// console (readable via `devicectl ... --console`) plus shows it on screen. The device numbers must
// match the Mac / HF reference (host DSP is deterministic).

// Same fixed series the CLI's --demo uses: 10 + 3·sin(2π i/48) + 0.01·i, i in 0..<512.
func demoSeries() -> [Float] {
    (0..<512).map { i in 10 + 3 * sinf(2 * .pi * Float(i) / 48) + 0.01 * Float(i) }
}

@MainActor
final class SelfTest: ObservableObject {
    @Published var status = "starting…"
    @Published var lines: [String] = []

    func run() async {
        do {
            status = "downloading model…"
            let forecaster = try await KitForecaster(catalog: "timesfm-2.5-200m") { p in
                Task { @MainActor in self.status = "downloading… \(Int(p.fraction * 100))%" }
            }
            status = "forecasting…"
            let series = demoSeries()
            let t0 = Date()
            _ = try await forecaster.forecast(series)                 // cold (specialize)
            let coldMs = Int(Date().timeIntervalSince(t0) * 1000)
            let t1 = Date()
            let f = try await forecaster.forecast(series)             // warm
            let ms = Int(Date().timeIntervalSince(t1) * 1000)
            print("GATE: cold=\(coldMs)ms warm=\(ms)ms")

            let mean8 = f.mean.prefix(8).map { String(format: "%.3f", $0) }.joined(separator: ", ")
            let p10 = (0..<4).map { String(format: "%.3f", f.quantiles[$0][1]) }.joined(separator: ", ")
            let p90 = (0..<4).map { String(format: "%.3f", f.quantiles[$0][9]) }.joined(separator: ", ")
            let gate = "GATE: mean[:8]=[\(mean8)] ms=\(ms)"
            print(gate)
            print("GATE: p10[:4]=[\(p10)] p90[:4]=[\(p90)]")
            status = "done (\(ms) ms)"
            lines = [gate, "p10[:4]=[\(p10)]", "p90[:4]=[\(p90)]",
                     "Mac ref mean[:8]=[12.536, 12.368, 12.236, 12.198, 12.163, 12.205, 12.284, 12.424]"]
        } catch {
            print("GATE: ERROR \(error)")
            status = "error: \(error)"
        }
    }
}

struct ContentView: View {
    @StateObject private var test = SelfTest()
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("TimesFM 2.5 — on-device self-test").font(.headline)
            Text(test.status).font(.subheadline).foregroundStyle(.secondary)
            ForEach(test.lines, id: \.self) { Text($0).font(.system(.caption, design: .monospaced)) }
            Spacer()
        }
        .padding()
        .task { await test.run() }
    }
}

@main
struct ForecastApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}
