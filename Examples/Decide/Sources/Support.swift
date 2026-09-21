// Support — the header, model picker and timing strip the three screens share.

import CoreAIOps
import SwiftUI

struct ScreenHeader: View {
    @Environment(DecideRuntime.self) private var runtime
    let title: String
    let subtitle: String

    var body: some View {
        @Bindable var runtime = runtime
        VStack(spacing: 4) {
            Text(title).font(.title2.bold())
            Text(subtitle).font(.caption).foregroundStyle(.secondary)
            HStack {
                Picker("Model", selection: $runtime.selectedID) {
                    ForEach(runtime.models) { entry in Text(entry.name).tag(entry.id) }
                }
                .pickerStyle(.menu)
                .disabled(runtime.status == .loading)
                Button(runtime.isReady && runtime.loadedID == runtime.selectedID ? "Loaded" : "Load") {
                    Task { await runtime.load() }
                }
                .disabled(runtime.status == .loading || (runtime.isReady && runtime.loadedID == runtime.selectedID))
            }
            Text(runtime.status.label).font(.callout).foregroundStyle(statusColor)
            if let f = runtime.downloadFraction {
                ProgressView(value: f).frame(maxWidth: 280)
            }
        }
    }

    private var statusColor: Color {
        switch runtime.status {
        case .error: return .red
        case .ready: return .green
        default: return .secondary
        }
    }
}

/// A probability as a short bar, so a table of decisions reads at a glance.
struct ProbabilityBar: View {
    let value: Double
    let tint: Color

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3).fill(.quaternary)
                RoundedRectangle(cornerRadius: 3).fill(tint)
                    .frame(width: max(2, geo.size.width * value))
            }
        }
        .frame(height: 8)
    }
}

func ms(_ value: Double) -> String { "\(Int(value.rounded())) ms" }

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    guard !sorted.isEmpty else { return 0 }
    return sorted.count % 2 == 1
        ? sorted[sorted.count / 2] : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
}
