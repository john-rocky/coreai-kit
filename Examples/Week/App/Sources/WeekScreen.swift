// WeekScreen — the whole app: one week planned by an on-device model. Top to bottom: the state pill
// (READY / ● PLANNING / DONE) with the clock from Plan (0.1 s steps) and the count planned; where
// the week comes from (your calendar, pasted lines, the sample week) and one line about it; one
// latency line; the spotlight, one event large enough to read in a small video (the latest answer
// with its chip; before a run, the first event with the question and the seven answers); the week,
// each event turning white with a chip for what it needs as its answer arrives, scrolled so the
// event being planned sits in the lower third; the seven answers as growing bars; Before your week,
// the events that need something; a small footer. With no week to plan, one card says why and
// offers the other sources. The look is TextClassify's InboxScreen (its colors, pill and line
// sizes, in units of width / 402).

import SwiftUI
import UniformTypeIdentifiers

enum WeekStyle {
    static let background = Color(rgb: 0x0E1116)
    static let lane = Color(rgb: 0x1B2028)
    static let pending = Color(rgb: 0x6B7280)
    static let dim = Color(rgb: 0x3C424C)
    static let axis = Color(rgb: 0x8A919C)
    static let latency = Color(rgb: 0xB8BEC6)
    static let action = Color(rgb: 0x4285F4)
    /// A line about the week that is not the plan: lines skipped, access off, nothing to plan.
    static let warn = Color(rgb: 0xFFB74D)

    static func badge(_ phase: WeekModel.Phase, detail: String) -> (text: String, color: Color) {
        switch phase {
        case .loading: return (detail.isEmpty ? "LOADING" : "LOADING \(detail)", Color(rgb: 0x5F6368))
        case .ready: return ("READY", Color(rgb: 0x5F6368))
        case .planning: return ("● PLANNING", Color(rgb: 0xE53935))
        case .done: return ("DONE", Color(rgb: 0x2E7D32))
        case .failed: return ("FAILED", Color(rgb: 0xE53935))
        }
    }

    /// Each answer keeps one color; nothing to prepare stays grey.
    static func color(_ bin: WeekBin) -> Color {
        switch bin {
        case .nothing: return axis
        case .document: return Color(rgb: 0x4285F4)
        case .prepare: return Color(rgb: 0xFBBC05)
        case .travel: return Color(rgb: 0xFF7043)
        case .online: return Color(rgb: 0x00ACC1)
        case .bring: return Color(rgb: 0x34A853)
        case .confirm: return Color(rgb: 0xAB47BC)
        }
    }
}

extension Color {
    /// 0xRRGGBB, sRGB.
    init(rgb: UInt32) {
        self.init(.sRGB, red: Double((rgb >> 16) & 0xFF) / 255, green: Double((rgb >> 8) & 0xFF) / 255,
                  blue: Double(rgb & 0xFF) / 255)
    }
}

extension WeekEvent {
    /// The line under an event's title, on a row and on the spotlight: the notes; the place when
    /// there are none.
    var notesLine: String {
        if let notes, !notes.isEmpty { return notes }
        return location ?? "no notes"
    }
}

struct WeekScreen: View {
    @Bindable var model: WeekModel

    var body: some View {
        GeometryReader { geo in
            let u = geo.size.width / 402
            VStack(alignment: .leading, spacing: 0) {
                WeekHeader(model: model, u: u)
                    .padding(.top, 8 * u)
                SourcePicker(model: model, u: u)
                    .padding(.top, 12 * u)
                Text(latencyLine)
                    .font(.system(size: 13.4 * u).monospacedDigit())
                    .foregroundStyle(WeekStyle.latency)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .padding(.top, 10 * u)
                if model.events.isEmpty {
                    EmptyWeek(model: model, u: u)
                        .padding(.top, 12 * u)
                    Spacer(minLength: 0)
                } else {
                    if let note = model.note {
                        Text(note)
                            .font(.system(size: 13 * u, weight: .semibold))
                            .foregroundStyle(WeekStyle.warn)
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)
                            .padding(.top, 4 * u)
                            .accessibilityIdentifier("week-note")
                    }
                    WeekSpotlight(model: model, u: u)
                        .padding(.top, 10 * u)
                    WeekList(model: model, u: u)
                        .padding(.top, 10 * u)
                    WeekPanel(model: model, u: u)
                        .padding(.top, 12 * u)
                }
                Text(footer)
                    .font(.system(size: 10.4 * u).monospacedDigit())
                    .foregroundStyle(WeekStyle.axis)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .padding(.top, 10 * u)
                    .padding(.bottom, 6 * u)
            }
            .padding(.horizontal, 16 * u)
        }
        .background(WeekStyle.background.ignoresSafeArea())
        .sheet(isPresented: $model.showingPaste) {
            PasteSheet(model: model)
        }
    }

    private var latencyLine: String {
        switch model.phase {
        case .loading:
            return model.readingWeek ? "loading the model · reading the week" : "\(model.count) events · loading the model"
        case .ready:
            if model.readingWeek { return "reading the week" }
            return model.events.isEmpty ? "nothing to plan yet" : "\(model.count) events · tap Plan my week"
        case .planning, .done:
            return WeekModel.ms(model.medianMs) + " per event · " + WeekModel.rate(model.rate) + " · int8 · Core AI GPU"
        case .failed: return model.detail
        }
    }

    private var footer: String {
        ["decider-0.8b int8", model.source.weekName, Device.os].joined(separator: " · ")
    }
}

/// Where the week comes from: three buttons, the current one filled. Paste opens the sheet again
/// when it is already picked; the calendar is read again. Off while planning.
struct SourcePicker: View {
    let model: WeekModel
    let u: CGFloat

    var body: some View {
        HStack(spacing: 6 * u) {
            ForEach(WeekModel.Source.allCases, id: \.self) { source in
                let picked = model.source == source
                Button {
                    Task { await model.select(source) }
                } label: {
                    Text(source.title)
                        .font(.system(size: 13 * u, weight: .semibold))
                        .foregroundStyle(picked ? .white : WeekStyle.latency)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7 * u)
                        .background(picked ? WeekStyle.action : WeekStyle.lane, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(model.phase == .planning)
                .accessibilityIdentifier("source-\(source.rawValue)")
            }
        }
        .opacity(model.phase == .planning ? 0.5 : 1)
    }
}

/// No week to plan: why (the calendar is being read, access is off, no event this week, nothing
/// pasted) and the ways to get one.
struct EmptyWeek: View {
    let model: WeekModel
    let u: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 14 * u) {
            Text(headline)
                .font(.system(size: 17 * u, weight: .semibold))
                .foregroundStyle(model.readingWeek ? .white : WeekStyle.warn)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("week-note")
            if !model.readingWeek {
                HStack(spacing: 10 * u) {
                    choice("Paste your week", primary: true) { Task { await model.select(.paste) } }
                    if model.source != .sample {
                        choice("Try the sample week", primary: false) { Task { await model.select(.sample) } }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 14 * u)
        .padding(.vertical, 16 * u)
        .background(WeekStyle.lane, in: RoundedRectangle(cornerRadius: 12 * u))
    }

    private var headline: String {
        if model.readingWeek { return model.source == .calendar ? "Reading this week from your calendar…" : "Reading…" }
        return model.note ?? "Nothing to plan"
    }

    private func choice(_ title: String, primary: Bool, _ run: @escaping () -> Void) -> some View {
        Button(action: run) {
            Text(title)
                .font(.system(size: 14 * u, weight: .semibold))
                .foregroundStyle(primary ? .white : WeekStyle.latency)
                .padding(.horizontal, 14 * u)
                .padding(.vertical, 9 * u)
                .background(primary ? WeekStyle.action : WeekStyle.background, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// Paste your week: one event a line, read as it is typed (how many events, how many lines
/// skipped, or why nothing can be planned), or a `week-cli dump --out` file imported as lines.
/// Plan puts the week on screen and plans it.
struct PasteSheet: View {
    @Bindable var model: WeekModel
    @State private var importing = false

    var body: some View {
        let reading = model.pasteReading
        let refused = model.pasteRefusal != nil || reading.problem != nil
        VStack(alignment: .leading, spacing: 10) {
            Text("Paste your week")
                .font(.title3.weight(.semibold))
            Text(verbatim: "One event a line: day, time, title · place · Notes: …\n"
                + "Mon 11:00 Lease renewal signing · the property office · Notes: bring two forms of ID")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            TextEditor(text: $model.pasteText)
                .font(.system(.footnote, design: .monospaced))
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(WeekStyle.lane, in: RoundedRectangle(cornerRadius: 8))
                .frame(minHeight: 280)
                .accessibilityIdentifier("paste-text")
            Text(model.pasteStatus)
                .font(.footnote.weight(.semibold).monospacedDigit())
                .foregroundStyle(refused ? WeekStyle.warn : WeekStyle.latency)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("paste-status")
            HStack(spacing: 10) {
                Button("Import .json…") { importing = true }
                Spacer()
                Button("Cancel") { model.showingPaste = false }
                    .keyboardShortcut(.cancelAction)
                Button(planTitle(reading)) { model.submitPaste() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(reading.problem != nil)
                    .accessibilityIdentifier("paste-plan")
            }
        }
        .padding(16)
        #if os(macOS)
        .frame(width: 400, height: 520)  // no wider than the phone-shaped window it hangs from
        #endif
        .background(WeekStyle.background)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.json, .plainText]) { result in
            if case .success(let url) = result { model.importFile(url) }
        }
        .onChange(of: model.pasteText) { model.pasteEdited() }
    }

    private func planTitle(_ reading: WeekInput.Reading) -> String {
        let n = reading.events.count
        guard reading.problem == nil, n > 0 else { return "Plan" }
        let events = "\(n) event\(n == 1 ? "" : "s")"
        return model.phase == .loading ? "Use \(events)" : "Plan \(events)"
    }
}

/// The pill, the clock and the count: the clock redraws ten times a second while planning.
struct WeekHeader: View {
    let model: WeekModel
    let u: CGFloat

    var body: some View {
        TimelineView(.animation(minimumInterval: 0.1, paused: model.phase != .planning)) { _ in
            let (badge, color) = WeekStyle.badge(model.phase, detail: model.detail)
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
                Text("\(model.done) / \(model.count)")
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

/// One event large enough to read in a small video: the event whose answer arrived last, with the
/// answer as a chip; before the first answer, the week's first event with the question and the
/// seven answers it can get. Every line keeps its height whatever it holds (the notes always take
/// two lines, the chip sits where the answers were), so the card never changes size and the list
/// under it never moves.
struct WeekSpotlight: View {
    let model: WeekModel
    let u: CGFloat

    var body: some View {
        let spot = model.spotlight
        VStack(alignment: .leading, spacing: 0) {
            Text(spot.map { "\($0.event.dayName) · \($0.event.time)" } ?? " ")
                .font(.system(size: 16 * u, weight: .semibold).monospacedDigit())
                .foregroundStyle(WeekStyle.axis)
            Text(spot?.event.title ?? " ")
                .font(.system(size: 21 * u, weight: .semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
                .padding(.top, 2 * u)
            // Two lines tall whatever the notes hold: a hidden two-line text sets the height (a
            // reserved second line comes out 2 pt shorter than a wrapped one).
            ZStack(alignment: .topLeading) {
                Text(verbatim: " \n ").hidden()
                Text(spot?.event.notesLine ?? " ").lineLimit(2)
            }
            .font(.system(size: 15 * u))
            .foregroundStyle(WeekStyle.latency)
            .padding(.top, 3 * u)
            Text(WeekQuestion.instructions)
                .font(.system(size: 13 * u, weight: .medium))
                .foregroundStyle(WeekStyle.axis)
                .lineLimit(1)
                .minimumScaleFactor(0.9)
                .padding(.top, 9 * u)
            ZStack(alignment: .topLeading) {
                Text(offered)
                    .font(.system(size: 14 * u, weight: .semibold))
                    .lineLimit(2, reservesSpace: true)
                    .opacity(spot?.result == nil ? 1 : 0)
                    .accessibilityHidden(spot?.result != nil)
                if let result = spot?.result { chip(result.bin) }
            }
            .padding(.top, 4 * u)
        }
        // Its full height always: squeezed by the screen, the notes would drop to one line.
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 14 * u)
        .padding(.vertical, 11 * u)
        .background(WeekStyle.lane, in: RoundedRectangle(cornerRadius: 12 * u))
    }

    /// The seven answers in plain words, `nothing` first, each in its bar's color; a line break
    /// never splits one.
    private var offered: AttributedString {
        var out = AttributedString()
        for bin in WeekBin.allCases {
            if !out.characters.isEmpty {
                var dot = AttributedString("\u{00A0}· ")
                dot.foregroundColor = WeekStyle.pending
                out += dot
            }
            var answer = AttributedString(Self.plainWords(bin).replacingOccurrences(of: " ", with: "\u{00A0}"))
            answer.foregroundColor = WeekStyle.color(bin)
            out += answer
        }
        return out
    }

    /// An answer as the card offers it before a run.
    static func plainWords(_ bin: WeekBin) -> String {
        switch bin {
        case .nothing: return "nothing to prepare"
        case .document: return "a document"
        case .prepare: return "prepare"
        case .travel: return "leave early"
        case .online: return "join online"
        case .bring: return "buy or bring"
        case .confirm: return "confirm"
        }
    }

    /// The answer, named as on the rows and the bars, in its bar's color.
    private func chip(_ bin: WeekBin) -> some View {
        Text(bin.label)
            .font(.system(size: 14 * u, weight: .bold))
            .foregroundStyle(WeekStyle.color(bin))
            .lineLimit(1)
            .fixedSize()
            .padding(.horizontal, 10 * u)
            .padding(.vertical, 4 * u)
            .background(WeekStyle.color(bin).opacity(0.16), in: RoundedRectangle(cornerRadius: 7 * u))
    }
}

/// The week. While planning it follows the event being planned, kept in the lower third.
struct WeekList: View {
    let model: WeekModel
    let u: CGFloat

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                LazyVStack(spacing: 2 * u) {
                    ForEach(model.events) { e in
                        WeekRow(
                            event: e, result: model.results[e.id],
                            current: model.phase == .planning && e.id == model.done, u: u
                        )
                        .id(e.id)
                    }
                }
                .padding(.bottom, 64 * u)  // room under the last event for the button
            }
            .scrollIndicators(.hidden)
            .onChange(of: model.done) { _, done in
                guard model.count > 0 else { return }
                withAnimation(.easeOut(duration: 0.25)) {
                    proxy.scrollTo(min(done, model.count - 1), anchor: UnitPoint(x: 0.5, y: 0.72))
                }
            }
            .overlay(alignment: .bottom) { button }
        }
    }

    @ViewBuilder private var button: some View {
        if model.phase == .ready {
            action("Plan my week", primary: true) { model.planWeek() }
        } else if model.phase == .done {
            if model.canAddReminders {
                let n = model.upcoming.count
                action("Add \(n) reminder\(n == 1 ? "" : "s")", primary: true) { Task { await model.addReminders() } }
            } else if model.addingReminders {
                action("Adding reminders…", primary: false) {}
            } else {
                action("Plan again", primary: false) { model.planWeek() }
            }
        }
    }

    private func action(_ title: String, primary: Bool, _ run: @escaping () -> Void) -> some View {
        Button(action: run) {
            Text(title)
                .font(.system(size: (primary ? 18 : 15) * u, weight: .semibold))
                .foregroundStyle(primary ? .white : WeekStyle.latency)
                .padding(.horizontal, (primary ? 34 : 24) * u)
                .padding(.vertical, (primary ? 13 : 10) * u)
                .background(primary ? WeekStyle.action : WeekStyle.lane, in: Capsule())
        }
        .buttonStyle(.plain)
        .padding(.bottom, 12 * u)
    }
}

/// One event: day and time, the title (grey until planned), a grey line from the notes, and the
/// answer as a chip. The height is the same before and after, so answers arriving never move the
/// list.
struct WeekRow: View {
    let event: WeekEvent
    let result: PlannedEvent?
    let current: Bool
    let u: CGFloat

    var body: some View {
        HStack(alignment: .top, spacing: 10 * u) {
            VStack(alignment: .leading, spacing: 2 * u) {
                Text(event.dayName)
                    .font(.system(size: 11 * u, weight: .bold))
                Text(event.time)
                    .font(.system(size: 11 * u, weight: .medium).monospacedDigit())
            }
            .foregroundStyle(result == nil && !current ? WeekStyle.dim : WeekStyle.axis)
            .frame(width: 38 * u, alignment: .leading)
            VStack(alignment: .leading, spacing: 3 * u) {
                HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
                    Text(event.title)
                        .font(.system(size: 13.5 * u, weight: .medium))
                        .foregroundStyle(result == nil ? WeekStyle.pending : .white)
                        .lineLimit(1)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if let result { chip(result.bin) }
                }
                .frame(height: 18 * u)
                Text(event.notesLine)
                    .font(.system(size: 11.5 * u))
                    .foregroundStyle(result == nil ? WeekStyle.dim : WeekStyle.pending)
                    .lineLimit(1)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.vertical, 6 * u)
        .padding(.horizontal, 8 * u)
        .background(current ? WeekStyle.lane : .clear, in: RoundedRectangle(cornerRadius: 8 * u))
    }

    private func chip(_ bin: WeekBin) -> some View {
        Text(bin.label)
            .font(.system(size: 10.4 * u, weight: .bold))
            .foregroundStyle(WeekStyle.color(bin))
            .lineLimit(1)
            .fixedSize()
            .padding(.horizontal, 7 * u)
            .padding(.vertical, 2.5 * u)
            .background(WeekStyle.lane, in: RoundedRectangle(cornerRadius: 6 * u))
    }
}

/// The counts: a DONE summary line (its row is kept while empty, so nothing moves), the seven
/// answers as bars, and Before your week — the events that need something, newest last.
struct WeekPanel: View {
    let model: WeekModel
    let u: CGFloat

    var body: some View {
        // Bars grow against a quarter of the week, or the largest count once one passes it.
        let scale = max(Double(model.count) / 4, Double(model.counts.values.max() ?? 0), 1)
        VStack(alignment: .leading, spacing: 5 * u) {
            Text(doneLine)
                .font(.system(size: 12.5 * u, weight: .semibold).monospacedDigit())
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .opacity(model.phase == .done ? 1 : 0)
            ForEach(WeekBin.allCases, id: \.self) { bin in
                bar(bin, count: model.counts[bin] ?? 0, scale: scale)
            }
            BeforeYourWeek(model: model, u: u)
                .padding(.top, 6 * u)
        }
    }

    private var doneLine: String {
        guard let run = model.run else { return " " }
        return "\(run.count) events · " + String(format: "%.1f s", run.totalSeconds) + " · median "
            + WeekModel.ms(model.medianMs) + " · " + model.source.weekName
    }

    private func bar(_ bin: WeekBin, count: Int, scale: Double) -> some View {
        let color = WeekStyle.color(bin)
        return HStack(spacing: 8 * u) {
            Text(bin.label)
                .font(.system(size: 11.5 * u, weight: .semibold))
                .foregroundStyle(color)
                .lineLimit(1)
                .frame(width: 96 * u, alignment: .leading)
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3 * u).fill(WeekStyle.lane)
                    RoundedRectangle(cornerRadius: 3 * u).fill(color)
                        .frame(width: g.size.width * min(1, Double(count) / scale))
                }
            }
            .frame(height: 9 * u)
            .animation(.linear(duration: 0.15), value: count)
            Text("\(count)")
                .font(.system(size: 11.5 * u, weight: .semibold).monospacedDigit())
                .foregroundStyle(.white)
                .frame(width: 28 * u, alignment: .trailing)
        }
        .frame(height: 14 * u)
    }
}

/// The plan: every event that needs something (day · title · what), growing as the answers
/// arrive, in a box that follows the newest line; the reminders' outcome in its heading.
struct BeforeYourWeek: View {
    let model: WeekModel
    let u: CGFloat

    var body: some View {
        let plan = model.plan
        VStack(alignment: .leading, spacing: 4 * u) {
            HStack(spacing: 0) {
                Text("Before your week").foregroundStyle(.white)
                Text(heading).foregroundStyle(WeekStyle.axis)
            }
            .font(.system(size: 12.5 * u, weight: .bold).monospacedDigit())
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    VStack(alignment: .leading, spacing: 2 * u) {
                        ForEach(plan, id: \.event.id) { item in
                            HStack(spacing: 6 * u) {
                                Text(item.event.dayName)
                                    .foregroundStyle(WeekStyle.axis)
                                    .frame(width: 30 * u, alignment: .leading)
                                Text(item.event.title)
                                    .foregroundStyle(.white)
                                    .lineLimit(1)
                                Spacer(minLength: 4 * u)
                                Text(item.bin.label)
                                    .foregroundStyle(WeekStyle.color(item.bin))
                                    .lineLimit(1)
                                    .fixedSize()
                            }
                            .font(.system(size: 11.5 * u, weight: .medium))
                            .frame(height: 15 * u)
                            .id(item.event.id)
                        }
                    }
                }
                .scrollIndicators(.hidden)
                .onChange(of: plan.count) { _, _ in
                    guard let last = plan.last else { return }
                    withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(last.event.id, anchor: .bottom) }
                }
            }
            .frame(height: 85 * u)
        }
    }

    private var heading: String {
        if let error = model.remindersError { return " · \(error)" }
        if let added = model.remindersAdded {
            var s = " · \(added) reminder\(added == 1 ? "" : "s") added"
            if model.remindersExisting > 0 { s += ", \(model.remindersExisting) already there" }
            return s
        }
        if model.addingReminders { return " · adding reminders" }
        let n = model.plan.count
        return n == 0 ? "" : " · \(n) to do"
    }
}
