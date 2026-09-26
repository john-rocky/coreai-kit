// InboxScreen — the whole app: one support inbox sorted by an on-device model. Top to bottom: the
// state pill (READY / ● SORTING / DONE) with the clock from Sort (0.1 s steps) and the count sorted;
// one latency line; the inbox, each message turning white with three chips (intent, urgency,
// sentiment) as its answers arrive, scrolled so the message being sorted sits in the lower third;
// the intent counts as growing bars and the critical / high count; a small footer. The colors, the
// pill and the line sizes follow the zoo's DiarizeLiveView (its sizes in units of width / 1080 are
// these in units of width / 402).

import SwiftUI

enum InboxStyle {
    static let background = Color(rgb: 0x0E1116)
    static let lane = Color(rgb: 0x1B2028)
    static let pending = Color(rgb: 0x6B7280)
    static let dim = Color(rgb: 0x3C424C)
    static let axis = Color(rgb: 0x8A919C)
    static let latency = Color(rgb: 0xB8BEC6)
    static let categories: [Color] = [0x4285F4, 0xEA4335, 0xFBBC05, 0x34A853, 0xAB47BC, 0x00ACC1, 0xFF7043, 0x9E9D24]
        .map { Color(rgb: $0) }
    static let action = Color(rgb: 0x4285F4)

    static func badge(_ phase: InboxModel.Phase, detail: String) -> (text: String, color: Color) {
        switch phase {
        case .loading: return (detail.isEmpty ? "LOADING" : "LOADING \(detail)", Color(rgb: 0x5F6368))
        case .ready: return ("READY", Color(rgb: 0x5F6368))
        case .sorting: return ("● SORTING", Color(rgb: 0xE53935))
        case .done: return ("DONE", Color(rgb: 0x2E7D32))
        case .failed: return ("FAILED", Color(rgb: 0xE53935))
        }
    }

    /// Each intent keeps one of the eight colors, in label order.
    static func intent(_ label: String) -> Color {
        categories[(InboxTasks.intent.labels.firstIndex(of: label) ?? 0) % categories.count]
    }

    static func urgency(_ label: String) -> Color {
        switch label {
        case "normal": return Color(rgb: 0x4285F4)
        case "high": return Color(rgb: 0xFF7043)
        case "critical": return Color(rgb: 0xE53935)
        default: return axis
        }
    }

    static func sentiment(_ label: String) -> Color {
        switch label {
        case "positive": return Color(rgb: 0x34A853)
        case "negative": return Color(rgb: 0xEA4335)
        default: return axis
        }
    }

    /// "order_status" -> "order status".
    static func display(_ label: String) -> String { label.replacingOccurrences(of: "_", with: " ") }
}

extension Color {
    /// 0xRRGGBB, sRGB.
    init(rgb: UInt32) {
        self.init(.sRGB, red: Double((rgb >> 16) & 0xFF) / 255, green: Double((rgb >> 8) & 0xFF) / 255,
                  blue: Double(rgb & 0xFF) / 255)
    }
}

/// 1,000 — the same grouping in every locale.
func grouped(_ n: Int) -> String {
    n.formatted(.number.locale(Locale(identifier: "en_US")))
}

struct InboxScreen: View {
    let model: InboxModel

    var body: some View {
        GeometryReader { geo in
            let u = geo.size.width / 402
            VStack(alignment: .leading, spacing: 0) {
                InboxHeader(model: model, u: u)
                    .padding(.top, 8 * u)
                Text(latencyLine)
                    .font(.system(size: 13.4 * u).monospacedDigit())
                    .foregroundStyle(InboxStyle.latency)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .padding(.top, 10 * u)
                InboxList(model: model, u: u)
                    .padding(.top, 12 * u)
                InboxPanel(model: model, u: u)
                    .padding(.top, 12 * u)
                Text(footer)
                    .font(.system(size: 10.4 * u).monospacedDigit())
                    .foregroundStyle(InboxStyle.axis)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .padding(.top, 10 * u)
                    .padding(.bottom, 6 * u)
            }
            .padding(.horizontal, 16 * u)
        }
        .background(InboxStyle.background.ignoresSafeArea())
    }

    private var latencyLine: String {
        switch model.phase {
        case .loading: return "\(grouped(model.count)) messages · loading the model"
        case .ready: return "\(grouped(model.count)) messages · tap Sort"
        case .sorting, .done:
            return InboxModel.ms(model.medianMs) + " per message · " + InboxModel.rate(model.rate)
                + " · \(model.precision) · Core AI GPU"
        case .failed: return model.detail
        }
    }

    /// The graphs that have run so far, with their message counts (a graph nothing ran on is left out).
    private var footer: String {
        let split = model.sequenceLengths.compactMap { S -> String? in
            guard let n = model.sequenceLengthCounts[S], n > 0 else { return nil }
            return "S=\(S): \(grouped(n))"
        }
        return (["gliner25-decide \(model.precision)"] + split + [Device.os]).joined(separator: " · ")
    }
}

/// The pill, the clock and the count: the clock redraws ten times a second while sorting.
struct InboxHeader: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        TimelineView(.animation(minimumInterval: 0.1, paused: model.phase != .sorting)) { _ in
            let (badge, color) = InboxStyle.badge(model.phase, detail: model.detail)
            HStack(alignment: .firstTextBaseline, spacing: 12 * u) {
                Text(badge)
                    .font(.system(size: 13.4 * u, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8 * u)
                    .padding(.vertical, 5 * u)
                    .background(color, in: Capsule())
                Text(clock)
                    .font(.system(size: 26.8 * u).monospacedDigit())
                    .foregroundStyle(.white)
                Spacer(minLength: 0)
                Text("\(grouped(model.done)) / \(grouped(model.count))")
                    .font(.system(size: 16.4 * u, weight: .bold).monospacedDigit())
                    .foregroundStyle(.white)
            }
        }
    }

    private var clock: String {
        guard let s = model.elapsed(at: .now) else { return "–.– s" }
        let tenths = Int(s * 10)
        return "\(tenths / 10).\(tenths % 10) s"
    }
}

/// The inbox. While sorting it follows the message being sorted, kept in the lower third.
struct InboxList: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                LazyVStack(spacing: 2 * u) {
                    ForEach(model.messages) { m in
                        InboxRow(
                            number: m.id + 1, text: m.text, result: model.results[m.id],
                            current: model.phase == .sorting && m.id == model.done, u: u
                        )
                        .id(m.id)
                    }
                }
                .padding(.bottom, 64 * u)  // room under the last message for the button
            }
            .scrollIndicators(.hidden)
            .onChange(of: model.done) { _, done in
                guard model.count > 0 else { return }
                // about one batch long, so the list glides instead of trailing the answers
                withAnimation(.linear(duration: 0.08)) {
                    proxy.scrollTo(min(done, model.count - 1), anchor: UnitPoint(x: 0.5, y: 0.72))
                }
            }
            .overlay(alignment: .bottom) { button }
        }
    }

    @ViewBuilder private var button: some View {
        if model.phase == .ready || model.phase == .done {
            let primary = model.phase == .ready
            Button {
                model.sort()
            } label: {
                Text(primary ? "Sort inbox" : "Sort again")
                    .font(.system(size: (primary ? 18 : 15) * u, weight: .semibold))
                    .foregroundStyle(primary ? .white : InboxStyle.latency)
                    .padding(.horizontal, (primary ? 34 : 24) * u)
                    .padding(.vertical, (primary ? 13 : 10) * u)
                    .background(primary ? InboxStyle.action : InboxStyle.lane, in: Capsule())
            }
            .buttonStyle(.plain)
            .padding(.bottom, 12 * u)
        }
    }
}

/// One message: its number, the text (grey until sorted), and the three answers as chips. The
/// height is the same before and after, so answers arriving never move the list.
struct InboxRow: View {
    let number: Int
    let text: String
    let result: SortedMessage?
    let current: Bool
    let u: CGFloat

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10 * u) {
            Text(String(format: "#%04d", number))
                .font(.system(size: 11 * u, weight: .medium).monospacedDigit())
                .foregroundStyle(result == nil && !current ? InboxStyle.dim : InboxStyle.axis)
            VStack(alignment: .leading, spacing: 5 * u) {
                Text(text)
                    .font(.system(size: 13 * u))
                    .foregroundStyle(result == nil ? InboxStyle.pending : .white)
                    .lineLimit(2, reservesSpace: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 6 * u) {
                    if let result {
                        chip(result.labels["intent"], InboxStyle.intent)
                        chip(result.labels["urgency"], InboxStyle.urgency)
                        chip(result.labels["sentiment"], InboxStyle.sentiment)
                    }
                }
                .frame(height: 18 * u, alignment: .leading)
            }
        }
        .padding(.vertical, 6 * u)
        .padding(.horizontal, 8 * u)
        .background(current ? InboxStyle.lane : .clear, in: RoundedRectangle(cornerRadius: 8 * u))
    }

    @ViewBuilder private func chip(_ label: String?, _ color: (String) -> Color) -> some View {
        if let label {
            Text(InboxStyle.display(label))
                .font(.system(size: 10.4 * u, weight: .bold))
                .foregroundStyle(color(label))
                .padding(.horizontal, 7 * u)
                .padding(.vertical, 2.5 * u)
                .background(InboxStyle.lane, in: RoundedRectangle(cornerRadius: 6 * u))
        }
    }
}

/// The counts: a DONE summary line (its row is kept while empty, so nothing moves), the eight
/// intents as bars, and how many messages were critical or high.
struct InboxPanel: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        let intents = model.counts["intent"] ?? [:]
        let urgency = model.counts["urgency"] ?? [:]
        // Bars grow against a quarter of the inbox, or the largest count once one passes it.
        let scale = max(Double(model.count) / 4, Double(intents.values.max() ?? 0), 1)
        VStack(alignment: .leading, spacing: 5 * u) {
            Text(doneLine)
                .font(.system(size: 12.5 * u, weight: .semibold).monospacedDigit())
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .opacity(model.phase == .done ? 1 : 0)
            ForEach(InboxTasks.intent.labels, id: \.self) { label in
                bar(label, count: intents[label] ?? 0, scale: scale)
            }
            HStack(spacing: 0) {
                Text("critical \(grouped(urgency["critical"] ?? 0))").foregroundStyle(InboxStyle.urgency("critical"))
                Text("  ·  ").foregroundStyle(InboxStyle.axis)
                Text("high \(grouped(urgency["high"] ?? 0))").foregroundStyle(InboxStyle.urgency("high"))
            }
            .font(.system(size: 12.5 * u, weight: .bold).monospacedDigit())
            .padding(.top, 3 * u)
        }
    }

    private var doneLine: String {
        guard let run = model.run else { return " " }
        return "\(grouped(run.count)) messages · \(InboxTasks.all.count) decisions each · "
            + String(format: "%.1f s", run.totalSeconds) + " · median " + InboxModel.ms(model.medianMs)
    }

    private func bar(_ label: String, count: Int, scale: Double) -> some View {
        let color = InboxStyle.intent(label)
        return HStack(spacing: 8 * u) {
            Text(InboxStyle.display(label))
                .font(.system(size: 11.5 * u, weight: .semibold))
                .foregroundStyle(color)
                .lineLimit(1)
                .frame(width: 132 * u, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3 * u).fill(InboxStyle.lane)
                    RoundedRectangle(cornerRadius: 3 * u).fill(color)
                        .frame(width: g.size.width * min(1, Double(count) / scale))
                }
            }
            .frame(height: 9 * u)
            .animation(.linear(duration: 0.1), value: count)
            Text(grouped(count))
                .font(.system(size: 11.5 * u, weight: .semibold).monospacedDigit())
                .foregroundStyle(.white)
                .frame(width: 36 * u, alignment: .trailing)
        }
        .frame(height: 15 * u)
    }
}
