// InboxScreen — the whole app: an inbox sorted by an on-device model into categories you name. Top
// to bottom: the state pill (READY / ● SORTING / DONE) with the clock from Sort (0.1 s steps) and
// the count sorted; where the inbox comes from (pasted text, a file, the sample inbox); the
// categories field; one latency line; Inbox or By category; the answers; the categories as growing
// bars and the critical / high count; a small footer. Inbox lists the messages in order, each
// turning white with three chips (category, urgency, sentiment) as its answers arrive, scrolled so
// the message being sorted sits in the lower third. By category lists every category with its
// messages, the most urgent first. With nothing to sort, one card says why and offers the other
// sources. The colors, the pill and the line sizes follow the zoo's DiarizeLiveView (its sizes in
// units of width / 1080 are these in units of width / 402).

import SwiftUI

enum InboxStyle {
    static let background = Color(rgb: 0x0E1116)
    static let lane = Color(rgb: 0x1B2028)
    static let pending = Color(rgb: 0x6B7280)
    static let dim = Color(rgb: 0x3C424C)
    static let axis = Color(rgb: 0x8A919C)
    static let latency = Color(rgb: 0xB8BEC6)
    /// One per category, in the field's order: the eight of the support inbox, then four more for a
    /// longer list of your own.
    static let categories: [Color] = [
        0x4285F4, 0xEA4335, 0xFBBC05, 0x34A853, 0xAB47BC, 0x00ACC1, 0xFF7043, 0x9E9D24,
        0xEC407A, 0x26A69A, 0x8D6E63, 0x7E57C2,
    ].map { Color(rgb: $0) }
    static let action = Color(rgb: 0x4285F4)
    /// A line about the inbox or the categories that is not an answer: skipped lines, a refusal.
    static let warn = Color(rgb: 0xFFB74D)

    static func badge(_ phase: InboxModel.Phase, detail: String) -> (text: String, color: Color) {
        switch phase {
        case .loading: return (detail.isEmpty ? "LOADING" : "LOADING \(detail)", Color(rgb: 0x5F6368))
        case .ready: return ("READY", Color(rgb: 0x5F6368))
        case .sorting: return ("● SORTING", Color(rgb: 0xE53935))
        case .done: return ("DONE", Color(rgb: 0x2E7D32))
        case .failed: return ("FAILED", Color(rgb: 0xE53935))
        }
    }

    /// Each category keeps one color, by its place in `categories`.
    static func category(_ label: String, in categories: [String]) -> Color {
        Self.categories[(categories.firstIndex(of: label) ?? 0) % Self.categories.count]
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

struct InboxScreen: View {
    @Bindable var model: InboxModel

    var body: some View {
        GeometryReader { geo in
            let u = geo.size.width / 402
            VStack(alignment: .leading, spacing: 0) {
                InboxHeader(model: model, u: u)
                    .padding(.top, 8 * u)
                SourcePicker(model: model, u: u)
                    .padding(.top, 12 * u)
                CategoriesField(model: model, u: u)
                    .padding(.top, 10 * u)
                Text(latencyLine)
                    .font(.system(size: 13.4 * u).monospacedDigit())
                    .foregroundStyle(InboxStyle.latency)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .padding(.top, 10 * u)
                if model.messages.isEmpty {
                    EmptyInbox(model: model, u: u)
                        .padding(.top, 12 * u)
                    Spacer(minLength: 0)
                } else {
                    if let note = model.note {
                        Text(note)
                            .font(.system(size: 13 * u, weight: .semibold))
                            .foregroundStyle(InboxStyle.warn)
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)
                            .padding(.top, 4 * u)
                            .accessibilityIdentifier("inbox-note")
                    }
                    ShowingPicker(model: model, u: u)
                        .padding(.top, 10 * u)
                    Group {
                        switch model.showing {
                        case .inbox: InboxList(model: model, u: u)
                        case .category: CategoryList(model: model, u: u)
                        }
                    }
                    .overlay(alignment: .bottom) { SortButton(model: model, u: u) }
                    .padding(.top, 8 * u)
                    InboxPanel(model: model, u: u)
                        .padding(.top, 12 * u)
                }
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
        .sheet(isPresented: $model.showingPaste) {
            PasteSheet(model: model)
        }
        .fileImporter(isPresented: $model.importing, allowedContentTypes: InboxModel.fileTypes) { result in
            if case .success(let url) = result { model.importFile(url) }
        }
        .onChange(of: model.categoriesText) { model.categoriesEdited() }
    }

    private var latencyLine: String {
        switch model.phase {
        case .loading: return "\(InboxModel.messagesLabel(model.count)) · loading the model"
        case .ready:
            if model.messages.isEmpty { return "nothing to sort yet" }
            if model.categories.problem != nil { return "\(InboxModel.messagesLabel(model.count)) · name the categories to sort" }
            return "\(InboxModel.messagesLabel(model.count)) · tap Sort inbox"
        case .sorting, .done:
            return InboxModel.ms(model.medianMs) + " per message · " + InboxModel.rate(model.rate)
                + " · \(model.precision) · Core AI GPU"
        case .failed: return model.detail
        }
    }

    /// The graphs that have run so far, with their message counts (a graph nothing ran on is left
    /// out), and where the inbox came from.
    private var footer: String {
        let split = model.sequenceLengths.compactMap { S -> String? in
            guard let n = model.sequenceLengthCounts[S], n > 0 else { return nil }
            return "S=\(S): \(grouped(n))"
        }
        return (["gliner25-decide \(model.precision)"] + split + [model.source.inboxName, Device.os]).joined(separator: " · ")
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

    /// Tenths, cut while the clock runs; rounded once done, so it reads as the DONE line does.
    private var clock: String {
        guard let s = model.elapsed(at: .now) else { return "–.– s" }
        let tenths = model.phase == .done ? Int((s * 10).rounded()) : Int(s * 10)
        return "\(tenths / 10).\(tenths % 10) s"
    }
}

/// A row of capsule buttons, the picked one filled.
struct Capsules<Item: Hashable>: View {
    let items: [Item]
    let picked: Item?
    let title: (Item) -> String
    let id: (Item) -> String
    let disabled: Bool
    let u: CGFloat
    let action: (Item) -> Void

    var body: some View {
        HStack(spacing: 6 * u) {
            ForEach(items, id: \.self) { item in
                let on = picked == item
                Button {
                    action(item)
                } label: {
                    Text(title(item))
                        .font(.system(size: 13 * u, weight: .semibold))
                        .foregroundStyle(on ? .white : InboxStyle.latency)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7 * u)
                        .background(on ? InboxStyle.action : InboxStyle.lane, in: Capsule())
                }
                .buttonStyle(.plain)
                .focusEffectDisabled()
                .disabled(disabled)
                .accessibilityIdentifier(id(item))
            }
        }
        .opacity(disabled ? 0.5 : 1)
    }
}

/// Where the inbox comes from: Paste opens the sheet, Import file the file picker, Sample inbox
/// puts the sample on screen. The inbox on screen keeps its source filled. Off while sorting.
struct SourcePicker: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        Capsules(
            items: InboxModel.Source.allCases, picked: model.source, title: \.title, id: { "source-\($0.rawValue)" },
            disabled: model.phase == .sorting, u: u
        ) { model.select($0) }
    }
}

/// The categories, separated by commas: the field, and why it cannot be sorted into when it
/// cannot. Off while sorting.
struct CategoriesField: View {
    @Bindable var model: InboxModel
    let u: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 4 * u) {
            HStack(spacing: 8 * u) {
                Text("Categories")
                    .font(.system(size: 12.5 * u, weight: .bold))
                    .foregroundStyle(InboxStyle.axis)
                TextField("work, family, bills", text: $model.categoriesText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13 * u, weight: .medium))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    #if os(iOS)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .submitLabel(.done)
                    #endif
                    .padding(.horizontal, 10 * u)
                    .padding(.vertical, 7 * u)
                    .background(InboxStyle.lane, in: RoundedRectangle(cornerRadius: 8 * u))
                    .disabled(model.phase == .sorting)
                    .accessibilityIdentifier("categories")
            }
            if let problem = model.categories.problem {
                Text(problem.message)
                    .font(.system(size: 12 * u, weight: .semibold))
                    .foregroundStyle(InboxStyle.warn)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("categories-problem")
            }
        }
    }
}

/// Inbox or By category.
struct ShowingPicker: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        Capsules(
            items: InboxModel.Showing.allCases, picked: model.showing, title: \.title, id: { "showing-\($0.rawValue)" },
            disabled: false, u: u
        ) { model.showing = $0 }
    }
}

/// No inbox to sort: why (nothing pasted, no file picked, a file that could not be read, nothing
/// readable) and the ways to get one.
struct EmptyInbox: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 14 * u) {
            Text(model.note ?? InboxInput.Problem.empty.message)
                .font(.system(size: 17 * u, weight: .semibold))
                .foregroundStyle(InboxStyle.warn)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("inbox-note")
            // One row when the three fit the width, else one under another.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10 * u) { choices }
                VStack(alignment: .leading, spacing: 8 * u) { choices }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 14 * u)
        .padding(.vertical, 16 * u)
        .background(InboxStyle.lane, in: RoundedRectangle(cornerRadius: 12 * u))
    }

    @ViewBuilder private var choices: some View {
        choice("Paste messages", primary: true) { model.select(.paste) }
        choice("Import a file", primary: false) { model.select(.file) }
        if model.source != .sample {
            choice("Sample inbox", primary: false) { model.select(.sample) }
        }
    }

    private func choice(_ title: String, primary: Bool, _ run: @escaping () -> Void) -> some View {
        Button(action: run) {
            Text(title)
                .font(.system(size: 14 * u, weight: .semibold))
                .foregroundStyle(primary ? .white : InboxStyle.latency)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .padding(.horizontal, 12 * u)
                .padding(.vertical, 9 * u)
                .background(primary ? InboxStyle.action : InboxStyle.background, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// Sort inbox (READY) or Sort again (DONE), over the bottom of the list. Greyed out while the
/// categories cannot be read, and then a press does nothing (`sort()` checks); not `.disabled`,
/// which macOS draws see-through over the list.
struct SortButton: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        if model.phase == .ready || model.phase == .done {
            let primary = model.phase == .ready && model.canSort
            Button {
                model.sort()
            } label: {
                Text(model.phase == .ready ? "Sort inbox" : "Sort again")
                    .font(.system(size: (model.phase == .ready ? 18 : 15) * u, weight: .semibold))
                    .foregroundStyle(primary ? .white : model.canSort ? InboxStyle.latency : InboxStyle.pending)
                    .padding(.horizontal, (model.phase == .ready ? 34 : 24) * u)
                    .padding(.vertical, (model.phase == .ready ? 13 : 10) * u)
                    .background(primary ? InboxStyle.action : InboxStyle.lane, in: Capsule())
            }
            .buttonStyle(.plain)
            .focusEffectDisabled()
            .padding(.bottom, 12 * u)
            .accessibilityIdentifier("sort")
        }
    }
}

/// The inbox. While sorting it follows the message being sorted, kept in the lower third.
struct InboxList: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        let categories = model.shownCategories
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                LazyVStack(spacing: 2 * u) {
                    ForEach(model.messages) { m in
                        InboxRow(
                            number: m.id + 1, text: m.text, result: model.results[m.id], categories: categories,
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
        }
    }
}

/// By category: every category with its count, then its messages, the most urgent first. Filled as
/// the answers arrive; before a run, the categories with nothing in them yet.
struct CategoryList: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        let groups = model.byCategory
        let categories = model.shownCategories
        ScrollView(.vertical) {
            LazyVStack(alignment: .leading, spacing: 2 * u) {
                if groups.isEmpty {
                    Text("Name at least two categories to sort into")
                        .font(.system(size: 13 * u, weight: .semibold))
                        .foregroundStyle(InboxStyle.warn)
                        .padding(.top, 8 * u)
                }
                ForEach(groups, id: \.category) { group in
                    let color = InboxStyle.category(group.category, in: categories)
                    HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
                        Circle().fill(color).frame(width: 8 * u, height: 8 * u)
                        Text(InboxStyle.display(group.category))
                            .font(.system(size: 14 * u, weight: .bold))
                            .foregroundStyle(color)
                            .lineLimit(1)
                        Spacer(minLength: 4 * u)
                        Text(InboxModel.messagesLabel(group.items.count))
                            .font(.system(size: 12 * u, weight: .semibold).monospacedDigit())
                            .foregroundStyle(group.items.isEmpty ? InboxStyle.dim : InboxStyle.axis)
                    }
                    .padding(.top, 10 * u)
                    .padding(.bottom, 2 * u)
                    .padding(.horizontal, 8 * u)
                    .accessibilityIdentifier("category-\(group.category)")
                    ForEach(group.items, id: \.message.id) { item in
                        CategoryRow(number: item.message.id + 1, text: item.message.text, result: item.result, u: u)
                    }
                }
            }
            .padding(.bottom, 64 * u)  // room under the last message for the button
        }
        .scrollIndicators(.hidden)
    }
}

/// One message under its category heading: the urgency first, in a column of equal chips (the order
/// the category is sorted in), the text in up to two lines, then the sentiment and the message's
/// number, and "truncated" under the urgency when the text was cut to fit the largest graph.
struct CategoryRow: View {
    let number: Int
    let text: String
    let result: SortedMessage
    let u: CGFloat

    var body: some View {
        let urgency = result.labels["urgency"] ?? ""
        let sentiment = result.labels["sentiment"] ?? ""
        HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
            VStack(alignment: .leading, spacing: 3 * u) {
                Text(urgency)
                    .font(.system(size: 10.4 * u, weight: .bold))
                    .foregroundStyle(InboxStyle.urgency(urgency))
                    .lineLimit(1)
                    .frame(width: 58 * u)
                    .padding(.vertical, 2.5 * u)
                    .background(InboxStyle.lane, in: RoundedRectangle(cornerRadius: 6 * u))
                if result.truncated {
                    Text("truncated")
                        .font(.system(size: 9.5 * u, weight: .semibold))
                        .foregroundStyle(InboxStyle.pending)
                        .accessibilityIdentifier("truncated")
                }
            }
            Text(text)
                .font(.system(size: 13 * u))
                .foregroundStyle(.white)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
            VStack(alignment: .trailing, spacing: 3 * u) {
                Text(sentiment)
                    .font(.system(size: 10.4 * u, weight: .bold))
                    .foregroundStyle(InboxStyle.sentiment(sentiment))
                Text(String(format: "#%04d", number))
                    .font(.system(size: 10 * u, weight: .medium).monospacedDigit())
                    .foregroundStyle(InboxStyle.dim)
            }
            .lineLimit(1)
            .fixedSize()
        }
        .padding(.vertical, 5 * u)
        .padding(.horizontal, 8 * u)
    }
}

/// One message: its number, the text (grey until sorted), and the answers as chips: the category,
/// the urgency, the sentiment, and "truncated" when the text was cut to fit the largest graph. The
/// height is the same before and after, so answers arriving never move the list.
struct InboxRow: View {
    let number: Int
    let text: String
    let result: SortedMessage?
    let categories: [String]
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
                        chip(result.labels["intent"]) { InboxStyle.category($0, in: categories) }
                        chip(result.labels["urgency"], InboxStyle.urgency)
                        chip(result.labels["sentiment"], InboxStyle.sentiment)
                        if result.truncated {
                            Text("truncated")
                                .font(.system(size: 9.5 * u, weight: .semibold))
                                .foregroundStyle(InboxStyle.pending)
                                .accessibilityIdentifier("truncated")
                        }
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
                .lineLimit(1)
                .padding(.horizontal, 7 * u)
                .padding(.vertical, 2.5 * u)
                .background(InboxStyle.lane, in: RoundedRectangle(cornerRadius: 6 * u))
        }
    }
}

/// The counts: a DONE summary line (its row is kept while empty, so nothing moves), the categories
/// as bars, and how many messages were critical or high.
struct InboxPanel: View {
    let model: InboxModel
    let u: CGFloat

    var body: some View {
        let categories = model.shownCategories
        let counts = model.counts["intent"] ?? [:]
        let urgency = model.counts["urgency"] ?? [:]
        // Bars grow against a quarter of the inbox, or the largest count once one passes it.
        let scale = max(Double(model.count) / 4, Double(counts.values.max() ?? 0), 1)
        // Twelve bars fit where eight did.
        let row: CGFloat = categories.count > 8 ? 12 : 15
        VStack(alignment: .leading, spacing: (categories.count > 8 ? 3 : 5) * u) {
            Text(doneLine)
                .font(.system(size: 12.5 * u, weight: .semibold).monospacedDigit())
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .opacity(model.phase == .done ? 1 : 0)
            ForEach(categories, id: \.self) { label in
                bar(label, color: InboxStyle.category(label, in: categories), count: counts[label] ?? 0, scale: scale, row: row)
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
        return "\(InboxModel.messagesLabel(run.count)) · \(run.tasks.count) decisions each · "
            + String(format: "%.1f s", run.totalSeconds) + " · median " + InboxModel.ms(model.medianMs)
    }

    private func bar(_ label: String, color: Color, count: Int, scale: Double, row: CGFloat) -> some View {
        HStack(spacing: 8 * u) {
            Text(InboxStyle.display(label))
                .font(.system(size: (row < 15 ? 10.5 : 11.5) * u, weight: .semibold))
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
            .frame(height: (row < 15 ? 7 : 9) * u)
            .animation(.linear(duration: 0.1), value: count)
            Text(grouped(count))
                .font(.system(size: 11.5 * u, weight: .semibold).monospacedDigit())
                .foregroundStyle(.white)
                .frame(width: 36 * u, alignment: .trailing)
        }
        .frame(height: row * u)
    }
}

/// Paste your messages: one a line, or blocks separated by a blank line, read as they are typed
/// (how many messages, how many skipped, or why nothing can be sorted). Sort puts the messages on
/// screen and sorts them into the categories on the main screen.
struct PasteSheet: View {
    @Bindable var model: InboxModel

    var body: some View {
        let reading = model.pasteReading
        let refused = model.pasteRefusal != nil || reading.problem != nil
        VStack(alignment: .leading, spacing: 10) {
            Text("Paste your messages")
                .font(.title3.weight(.semibold))
            Text("One message a line. For messages that run over several lines, put a blank line between them.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            TextEditor(text: $model.pasteText)
                .font(.system(.footnote))
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(InboxStyle.lane, in: RoundedRectangle(cornerRadius: 8))
                .frame(minHeight: 280)
                .accessibilityIdentifier("paste-text")
            Text(model.pasteStatus)
                .font(.footnote.weight(.semibold).monospacedDigit())
                .foregroundStyle(refused ? InboxStyle.warn : InboxStyle.latency)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("paste-status")
            HStack(spacing: 10) {
                // The system's Paste button reads the clipboard on a tap, without the alert a
                // program reading the clipboard by itself gets.
                PasteButton(payloadType: String.self) { strings in
                    let text = strings.joined(separator: "\n")
                    Task { @MainActor in model.pasteText = text }
                }
                .labelStyle(.titleAndIcon)
                Spacer()
                Button("Cancel") { model.showingPaste = false }
                    .keyboardShortcut(.cancelAction)
                Button(sortTitle(reading)) { model.submitPaste() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(reading.problem != nil)
                    .accessibilityIdentifier("paste-sort")
            }
        }
        .padding(16)
        #if os(macOS)
        .frame(width: 400, height: 520)  // no wider than the phone-shaped window it hangs from
        #endif
        .background(InboxStyle.background)
        .onChange(of: model.pasteText) { model.pasteEdited() }
    }

    private func sortTitle(_ reading: InboxInput.Reading) -> String {
        let n = reading.messages.count
        guard reading.problem == nil, n > 0 else { return "Sort" }
        return model.phase == .loading ? "Use \(InboxModel.messagesLabel(n))" : "Sort \(InboxModel.messagesLabel(n))"
    }
}
