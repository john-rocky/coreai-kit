// DemoShell — the visual shell shared by the Wave 2.5 demo apps (Spotlight / Siri / Visual
// Intelligence): one identity header, one always-on "your model · on-device · offline" badge, the
// honest "YOUR model, not Apple's" line, and a process-trace row. Keeping these here (instead of
// inline in each screen) is what makes the three demos look like one brand. Pure SwiftUI, no kit
// types — copy this file into a sibling Example to adopt the same shell.

import SwiftUI

/// Brand constants for the demo. The model name is shown verbatim in the badge and copy so the
/// "your own model on device" claim is always on screen next to the answer it produced.
enum Demo {
    static let modelName = "Gemma 4 E2B"
    /// The app's display name (CFBundleDisplayName) — also what `\(.applicationName)` resolves to,
    /// so the Siri phrase is "Ask Gemma".
    static let appName = "Gemma"
    /// Cold first-run download of the Gemma 4 E2B bundle + PLE tables (honest, shown — not hidden).
    static let modelDownloadHint = "~4.4 GB, one time"
    static let accent = Color.indigo
}

// MARK: - Identity header (shell ①②③ + model status)

/// The top of every demo screen: what this is, in one line, plus the always-on badge and the
/// honest one-liner. Folds in the live model status so the user never sees a dead/empty header
/// (legibility bar #1–#3 + #5).
struct IdentityHeader: View {
    let title: String
    let subtitle: String
    var status: ModelStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.title2.weight(.bold))
                    Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
            }
            HStack(spacing: 8) {
                DemoBadge()
                NotApplePill()
            }
            ModelStatusLine(status: status)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The always-on badge: `Gemma 4 E2B · on-device · ✈️ offline`. Identical wording across the demos
/// = the shared brand mark (shell ②).
struct DemoBadge: View {
    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "cpu").imageScale(.small)
            Text(Demo.modelName).fontWeight(.semibold)
            Text("·").foregroundStyle(.secondary)
            Text("on-device")
            Text("·").foregroundStyle(.secondary)
            Image(systemName: "airplane").imageScale(.small)
            Text("offline")
        }
        .font(.caption.weight(.medium))
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.green.opacity(0.15), in: Capsule())
        .foregroundStyle(.green)
    }
}

/// The honest answer to "why not just Apple Intelligence?" (shell ③ / legibility #5).
struct NotApplePill: View {
    var body: some View {
        Text("YOUR model, not Apple's")
            .font(.caption.weight(.medium))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Color.secondary.opacity(0.12), in: Capsule())
            .foregroundStyle(.secondary)
    }
}

/// Live model state under the header, so the cold multi-GB first-run download is visible (honest)
/// and "Ready" tells the user the on-device model is warm for the next Siri call.
struct ModelStatusLine: View {
    var status: ModelStatus

    var body: some View {
        Group {
            switch status.phase {
            case .ready:
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
                    Text("Ready — warm for Siri")
                    if let s = status.loadSeconds {
                        Text(String(format: "· loaded %.1fs", s)).foregroundStyle(.secondary)
                    }
                }
            case .error(let message):
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange).lineLimit(2)
            case .downloading(let fraction):
                HStack(spacing: 8) {
                    ProgressView(value: fraction).frame(width: 110)
                    Text("Downloading \(Demo.modelName)… \(Int(fraction * 100))%  (\(Demo.modelDownloadHint))")
                        .foregroundStyle(.secondary)
                }
            default:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading \(Demo.modelName) on device…").foregroundStyle(.secondary)
                }
            }
        }
        .font(.caption.monospacedDigit())
    }
}

// MARK: - Card container

/// The rounded card every section sits in — the shared container that makes the screens feel like
/// one app. `accent` tints the leading edge.
struct DemoCard<Content: View>: View {
    var accent: Color = .secondary
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            // Cross-platform card fill (no UIColor/NSColor symbols → builds on iOS + macOS).
            .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 16))
            .overlay(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(accent)
                    .frame(width: 4)
                    .padding(.vertical, 12)
            }
    }
}

/// A small caps section label above a group of cards.
struct SectionLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text.uppercased())
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 4)
    }
}

// MARK: - Process trace (shell ④ "make the magic visible")

/// One step of the on-device pipeline (asked → answered), shown live so the user sees the model
/// actually working rather than a black-box spinner.
struct TraceRow: View {
    let line: TraceLine

    var body: some View {
        HStack(spacing: 8) {
            if line.done {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green).imageScale(.small)
            } else if line.failed {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.orange).imageScale(.small)
            } else {
                ProgressView().controlSize(.small)
            }
            Image(systemName: line.icon).foregroundStyle(.secondary).frame(width: 16)
            Text(line.text).foregroundStyle(line.done ? .primary : .secondary)
            Spacer(minLength: 0)
        }
        .font(.caption)
    }
}
