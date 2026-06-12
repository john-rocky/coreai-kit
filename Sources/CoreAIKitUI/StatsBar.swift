// StatsBar.swift — compact live-performance footer for GenerationStats.

import SwiftUI

public struct StatsBar: View {
    private let stats: GenerationStats

    public init(stats: GenerationStats) {
        self.stats = stats
    }

    public var body: some View {
        HStack(spacing: 14) {
            if let load = stats.loadSeconds {
                item("load", String(format: "%.1fs", load))
            }
            if let ttft = stats.ttftSeconds {
                item("TTFT", String(format: "%.2fs", ttft))
            }
            if let tps = stats.tokensPerSecond {
                item("tok/s", String(format: "%.1f", tps))
            }
            item("tokens", "\(stats.promptTokens)→\(stats.generatedTokens)")
            item(
                "mem",
                ByteCountFormatter.string(
                    fromByteCount: Int64(stats.footprintBytes), countStyle: .memory))
            Spacer()
        }
        .font(.caption.monospacedDigit())
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func item(_ label: String, _ value: String) -> some View {
        HStack(spacing: 3) {
            Text(label).foregroundStyle(.secondary)
            Text(value)
        }
    }
}
