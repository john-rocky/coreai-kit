// RoomCheckView — the Room check screen, dark and full height, on a 9:16 column (a wider window gets bands at the
// sides), with its own header: `ScreenHeader` carries the text model's picker, and this screen never loads a text
// model. Top to bottom: the state pill (READY / ● CHECKING / DONE) with the clock from the press (0.1 s steps) and
// the rooms done; the rooms as a 2 × 3 grid, each with its verdict once answered, and Check all; while checking, the
// room last answered large with its verdict and five rows (the first room shows while its pass runs) over a strip of
// the rooms; at DONE the grid again, the rows to check by hand, the count line and the latency line; a small footer.
// Tap an answered room, or a row to check, to open it. Colors and sizes follow the TextClassify inbox demo (units of
// width / 402), so a phone recording reads at feed width. Every number comes from `RoomCheckModel`'s run.

import CoreAIOps
import SwiftUI

enum RoomStyle {
    static let background = Color(rgb: 0x0E1116)
    static let lane = Color(rgb: 0x1B2028)
    static let pending = Color(rgb: 0x6B7280)
    static let axis = Color(rgb: 0x8A919C)
    static let latency = Color(rgb: 0xB8BEC6)
    static let action = Color(rgb: 0x4285F4)

    static func pill(_ phase: RoomCheckModel.Phase, detail: String) -> (text: String, color: Color) {
        switch phase {
        case .loading: return (detail.isEmpty ? "LOADING" : "LOADING \(detail)", Color(rgb: 0x5F6368))
        case .ready: return ("READY", Color(rgb: 0x5F6368))
        case .checking: return ("● CHECKING", Color(rgb: 0xE53935))
        case .done: return ("DONE", Color(rgb: 0x2E7D32))
        case .failed: return ("FAILED", Color(rgb: 0xE53935))
        }
    }

    /// The verdict badge: its fill and its text color.
    static func verdict(_ verdict: RoomCheckModel.Verdict) -> (fill: Color, text: Color) {
        switch verdict {
        case .ready: return (Color(rgb: 0x2E7D32), .white)
        case .reclean: return (Color(rgb: 0xE53935), .white)
        case .check: return (Color(rgb: 0xFBBC05), Color(rgb: 0x0E1116))
        }
    }

    /// The verdict in one character, for the strip's small tiles.
    static func glyph(_ verdict: RoomCheckModel.Verdict) -> String {
        switch verdict {
        case .ready: return "✓"
        case .reclean: return "✗"
        case .check: return "?"
        }
    }

    static func flag(_ flag: RoomCheckModel.Flag) -> Color {
        switch flag {
        case .ok: return Color(rgb: 0x34A853)
        case .miss: return Color(rgb: 0xEA4335)
        case .check: return Color(rgb: 0xFBBC05)
        }
    }
}

extension Color {
    /// 0xRRGGBB, sRGB.
    init(rgb: UInt32) {
        self.init(
            .sRGB, red: Double((rgb >> 16) & 0xFF) / 255, green: Double((rgb >> 8) & 0xFF) / 255,
            blue: Double(rgb & 0xFF) / 255)
    }
}

struct RoomCheckView: View {
    @Environment(Autoplay.self) private var autoplay
    @State private var model = RoomCheckModel()

    var body: some View {
        GeometryReader { geo in
            let width = min(geo.size.width, geo.size.height * 9 / 16)
            RoomColumn(model: model, u: width / 402)
                .frame(width: width, height: geo.size.height)
                .frame(maxWidth: .infinity)
        }
        .background(RoomStyle.background.ignoresSafeArea())
        .environment(\.colorScheme, .dark)
        .task {
            // Returns at once unless this screen is the autoplayed one.
            await autoplay.run(
                .room, status: { model.statusLine }, finished: { model.isFinished },
                action: { model.checkAll() },
                load: {
                    let t0 = Date()
                    await model.load()
                    let ms = Int(Date().timeIntervalSince(t0) * 1000)
                    let state = model.phase == .ready ? "Ready" : "Error: \(model.detail)"
                    return ("load \(state) in \(ms) ms · \(RoomCheckModel.modelID) \(model.grid)", model.phase == .ready)
                })
            await model.load()
            if autoplay.screen == .room, let n = autoplay.detail, model.phase == .done, n <= model.count {
                try? await Task.sleep(for: .seconds(3))
                model.selected = n - 1
            }
        }
    }
}

/// The whole screen at one width: `u` is a point at 402 wide.
struct RoomColumn: View {
    let model: RoomCheckModel
    let u: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            RoomHeader(model: model, u: u)
                .padding(.top, 8 * u)
            Text("Room check · 5 checks per photo")
                .font(.system(size: 14 * u, weight: .semibold))
                .foregroundStyle(RoomStyle.latency)
                .padding(.top, 6 * u)
            content
                .padding(.top, 10 * u)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .overlay { detail }
            Text(doneLine)
                .font(.system(size: 12.5 * u, weight: .semibold).monospacedDigit())
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .opacity(model.phase == .done ? 1 : 0)
                .padding(.top, 8 * u)
            Text(latencyLine)
                .font(.system(size: 13.4 * u).monospacedDigit())
                .foregroundStyle(RoomStyle.latency)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .padding(.top, 4 * u)
            Text(footer)
                .font(.system(size: 10.4 * u).monospacedDigit())
                .foregroundStyle(RoomStyle.axis)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .padding(.top, 8 * u)
                .padding(.bottom, 6 * u)
        }
        .padding(.horizontal, 16 * u)
    }

    @ViewBuilder private var content: some View {
        switch model.phase {
        case .checking:
            VStack(spacing: 10 * u) {
                if let index = model.shown, index < model.count {
                    RoomPanel(room: model.rooms[index], number: index + 1, result: model.results[index], u: u)
                }
                RoomStrip(model: model, u: u)
            }
        default:
            ViewThatFits(in: .vertical) {
                VStack(alignment: .leading, spacing: 12 * u) {
                    rooms
                    button
                }
                VStack(spacing: 12 * u) {
                    ScrollView(.vertical) { rooms }
                        .scrollIndicators(.hidden)
                    button
                }
            }
        }
    }

    private var rooms: some View {
        VStack(alignment: .leading, spacing: 12 * u) {
            RoomGrid(model: model, u: u)
            if model.phase == .done { HandCheckList(model: model, u: u) }
        }
    }

    @ViewBuilder private var button: some View {
        if model.phase == .ready || model.phase == .done {
            let primary = model.phase == .ready
            Button {
                model.checkAll()
            } label: {
                Text(primary ? "Check all" : "Check again")
                    .font(.system(size: (primary ? 18 : 15) * u, weight: .semibold))
                    .foregroundStyle(primary ? .white : RoomStyle.latency)
                    .padding(.horizontal, (primary ? 34 : 24) * u)
                    .padding(.vertical, (primary ? 13 : 10) * u)
                    .background(primary ? RoomStyle.action : RoomStyle.lane, in: Capsule())
            }
            .buttonStyle(.plain)
            .frame(maxWidth: .infinity)
        }
    }

    /// The room opened by a tap, over the content.
    @ViewBuilder private var detail: some View {
        if let index = model.selected, index < model.count, let result = model.results[index] {
            let room = model.rooms[index]
            VStack(alignment: .leading, spacing: 8 * u) {
                HStack {
                    Text(room.name)
                        .font(.system(size: 11.5 * u).monospacedDigit())
                        .foregroundStyle(RoomStyle.axis)
                    Spacer()
                    Button {
                        model.selected = nil
                    } label: {
                        Text("Close")
                            .font(.system(size: 13 * u, weight: .semibold))
                            .foregroundStyle(RoomStyle.latency)
                            .padding(.horizontal, 12 * u)
                            .padding(.vertical, 5 * u)
                            .background(RoomStyle.lane, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
                RoomPanel(room: room, number: index + 1, result: result, u: u)
                Text(internals(result))
                    .font(.system(size: 10.4 * u).monospacedDigit())
                    .foregroundStyle(RoomStyle.axis)
                    .lineLimit(2)
                    .minimumScaleFactor(0.7)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .background(RoomStyle.background)
            .contentShape(Rectangle())
            .onTapGesture { model.selected = nil }
        }
    }

    /// What the kit reports about the pass: the prompt's tokens, the decoder's calls and the stages' seconds.
    private func internals(_ r: RoomCheckModel.Result) -> String {
        func s(_ key: String) -> String { String(format: "%.2f s", r.stages[key] ?? 0) }
        let image = (r.stages["decode"] ?? 0) + (r.stages["resize"] ?? 0) + (r.stages["patches"] ?? 0)
        return "\(r.tokens) tokens · \(r.prefillCalls) prefill + \(r.mainCalls) single-token decoder calls · "
            + String(format: "image %.2f s · ", image) + "tower \(s("tower")) · decoder \(s("decoder"))"
            + String(format: " · %.2f s on this screen's clock", r.seconds)
    }

    private var doneLine: String {
        guard let t = model.totals else { return " " }
        return "\(t.rooms) rooms · \(t.checks) checks · \(t.ready) ready · \(t.reclean) reclean · "
            + "\(t.checkRows) to check by hand"
    }

    private var perRoom: String {
        "\(RoomCheckModel.checks.count) questions · 1 pass · on-device"
    }

    private var latencyLine: String {
        switch model.phase {
        case .loading: return "\(RoomCheckModel.modelID) · loading the model"
        case .ready: return "\(model.count) rooms · tap Check all"
        case .checking:
            guard let m = model.medianSoFar else { return perRoom }
            return String(format: "%.1f s per room · ", m) + perRoom
        case .done:
            guard let t = model.totals else { return perRoom }
            return String(format: "%.1f s per room · ", t.medianRoomSeconds) + perRoom
        case .failed: return model.detail
        }
    }

    private var footer: String {
        "\(RoomCheckModel.modelID) (Mapika, Apache-2.0) · Core AI · \(model.grid) · \(RoomCheckModel.deviceName)"
    }
}

/// The pill, the clock and the rooms done: the clock redraws ten times a second while checking.
struct RoomHeader: View {
    let model: RoomCheckModel
    let u: CGFloat

    var body: some View {
        TimelineView(.animation(minimumInterval: 0.1, paused: model.phase != .checking)) { context in
            let (text, color) = RoomStyle.pill(model.phase, detail: model.detail)
            HStack(alignment: .firstTextBaseline, spacing: 12 * u) {
                Text(text)
                    .font(.system(size: 13.4 * u, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8 * u)
                    .padding(.vertical, 5 * u)
                    .background(color, in: Capsule())
                Text(clock(at: context.date))
                    .font(.system(size: 26.8 * u).monospacedDigit())
                    .foregroundStyle(.white)
                Spacer(minLength: 0)
                Text("\(model.doneCount) / \(model.count) rooms")
                    .font(.system(size: 16.4 * u, weight: .bold).monospacedDigit())
                    .foregroundStyle(.white)
            }
        }
    }

    /// While checking, the tenths elapsed; at DONE, the run's total as recorded, rounded like every other number.
    private func clock(at now: Date) -> String {
        guard let s = model.elapsed(at: now) else { return "0.0 s" }
        if model.phase == .done { return String(format: "%.1f s", s) }
        let tenths = Int(s * 10)
        return "\(tenths / 10).\(tenths % 10) s"
    }
}

/// The rooms as a grid of three columns (2 × 3 for six), each with its verdict once answered.
struct RoomGrid: View {
    let model: RoomCheckModel
    let u: CGFloat

    var body: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8 * u), count: 3), spacing: 8 * u) {
            ForEach(model.rooms) { room in
                RoomTile(room: room, number: room.id + 1, verdict: model.results[room.id]?.verdict, mode: .shown, u: u)
                    .onTapGesture { if model.results[room.id] != nil { model.selected = room.id } }
            }
        }
    }
}

/// The rooms in a line while checking: the one whose pass runs outlined, the ones to come dimmed.
struct RoomStrip: View {
    let model: RoomCheckModel
    let u: CGFloat

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal) {
                HStack(spacing: 6 * u) {
                    ForEach(model.rooms) { room in
                        let mode: RoomTile.Mode =
                            room.id == model.checking ? .running : model.results[room.id] == nil ? .waiting : .shown
                        RoomTile(
                            room: room, number: room.id + 1, verdict: model.results[room.id]?.verdict, mode: mode,
                            small: true, u: u
                        )
                        .frame(width: 56 * u)
                        .id(room.id)
                    }
                }
            }
            .scrollIndicators(.hidden)
            .onChange(of: model.checking) { _, index in
                guard let index else { return }
                withAnimation(.linear(duration: 0.15)) { proxy.scrollTo(index, anchor: .center) }
            }
        }
        .frame(height: 56 * u)
    }
}

struct RoomTile: View {
    enum Mode { case shown, running, waiting }

    let room: RoomCheckModel.Room
    let number: Int
    let verdict: RoomCheckModel.Verdict?
    let mode: Mode
    var small = false
    let u: CGFloat

    var body: some View {
        RoomStyle.lane
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if let image = room.image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFill()
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8 * u))
            .opacity(mode == .waiting ? 0.4 : 1)
            .overlay {
                if mode == .running {
                    RoundedRectangle(cornerRadius: 8 * u).stroke(.white, lineWidth: 2 * u)
                }
            }
            .overlay(alignment: .topLeading) {
                if !small {
                    Text("\(number)")
                        .font(.system(size: 11 * u, weight: .bold).monospacedDigit())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6 * u)
                        .padding(.vertical, 2 * u)
                        .background(.black.opacity(0.55), in: Capsule())
                        .padding(5 * u)
                }
            }
            .overlay(alignment: small ? .bottom : .bottomLeading) {
                if let verdict {
                    let (fill, text) = RoomStyle.verdict(verdict)
                    Text(small ? RoomStyle.glyph(verdict) : verdict.rawValue)
                        .font(.system(size: (small ? 10 : 11.5) * u, weight: .heavy))
                        .foregroundStyle(text)
                        .padding(.horizontal, (small ? 5 : 7) * u)
                        .padding(.vertical, 2.5 * u)
                        .background(fill, in: Capsule())
                        .padding((small ? 3 : 5) * u)
                }
            }
    }
}

/// One room large: the photo, the verdict and the pass's seconds, the five rows. Before its pass ends the rows wait
/// with their labels only; they arrive together.
struct RoomPanel: View {
    let room: RoomCheckModel.Room
    let number: Int
    let result: RoomCheckModel.Result?
    let u: CGFloat

    var body: some View {
        VStack(alignment: .leading, spacing: 8 * u) {
            ZStack {
                if let image = room.image {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 10 * u))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            HStack(alignment: .firstTextBaseline, spacing: 10 * u) {
                Text("Room \(number)")
                    .font(.system(size: 16 * u, weight: .bold))
                    .foregroundStyle(.white)
                if let result {
                    let (fill, text) = RoomStyle.verdict(result.verdict)
                    Text(result.verdict.rawValue)
                        .font(.system(size: 15 * u, weight: .heavy))
                        .foregroundStyle(text)
                        .padding(.horizontal, 10 * u)
                        .padding(.vertical, 3 * u)
                        .background(fill, in: Capsule())
                    Spacer(minLength: 0)
                    Text(String(format: "%.2f s", result.seconds))
                        .font(.system(size: 14 * u).monospacedDigit())
                        .foregroundStyle(RoomStyle.latency)
                } else {
                    Text("checking…")
                        .font(.system(size: 15 * u, weight: .semibold))
                        .foregroundStyle(RoomStyle.pending)
                    Spacer(minLength: 0)
                }
            }
            VStack(spacing: 4 * u) {
                ForEach(Array(RoomCheckModel.checks.enumerated()), id: \.offset) { k, check in
                    RoomRow(check: check, row: result?.rows[k], u: u)
                }
            }
        }
    }
}

/// One check: the mark (✓ the clean-room answer, ✗ not), the question in short, the answer and its probability, a
/// bar; below 0.8 the row is yellow and goes to a person.
struct RoomRow: View {
    let check: RoomCheckModel.Check
    let row: RoomCheckModel.Row?
    let u: CGFloat

    var body: some View {
        let color = row.map { RoomStyle.flag($0.flag) } ?? RoomStyle.pending
        VStack(alignment: .leading, spacing: 4 * u) {
            HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
                Text(mark)
                    .font(.system(size: 16 * u, weight: .heavy))
                    .foregroundStyle(color)
                    .frame(width: 16 * u, alignment: .center)
                Text(check.label)
                    .font(.system(size: 15 * u, weight: .semibold))
                    .foregroundStyle(row == nil ? RoomStyle.pending : .white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                Spacer(minLength: 4 * u)
                if let row {
                    Text(row.answer)
                        .font(.system(size: 15 * u, weight: .bold))
                        .foregroundStyle(.white)
                    Text(String(format: "%.2f", row.p))
                        .font(.system(size: 15 * u, weight: .semibold).monospacedDigit())
                        .foregroundStyle(color)
                        .frame(width: 40 * u, alignment: .trailing)
                }
            }
            HStack(spacing: 8 * u) {
                Color.clear.frame(width: 16 * u, height: 1)
                GeometryReader { g in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3 * u).fill(RoomStyle.background)
                        RoundedRectangle(cornerRadius: 3 * u).fill(color)
                            .frame(width: g.size.width * (row?.p ?? 0))
                    }
                }
                .frame(height: 7 * u)
                if row?.flag == .check {
                    Text("check by hand")
                        .font(.system(size: 11.5 * u, weight: .bold))
                        .foregroundStyle(color)
                }
            }
            .frame(height: 13 * u)
        }
        .padding(.vertical, 5 * u)
        .padding(.horizontal, 8 * u)
        .background(RoomStyle.lane, in: RoundedRectangle(cornerRadius: 8 * u))
    }

    private var mark: String {
        guard let row else { return "·" }
        return row.answer == check.expected ? "✓" : "✗"
    }
}

/// At DONE: every row below 0.8, by room — tap one to open its room.
struct HandCheckList: View {
    let model: RoomCheckModel
    let u: CGFloat

    var body: some View {
        let flagged = model.rooms.flatMap { room in
            (model.results[room.id]?.rows ?? []).filter { $0.flag == .check }.map { (key: "\(room.id)-\($0.id)", room: room, row: $0) }
        }
        if !flagged.isEmpty {
            VStack(alignment: .leading, spacing: 4 * u) {
                Text("To check by hand")
                    .font(.system(size: 11.5 * u, weight: .semibold))
                    .foregroundStyle(RoomStyle.axis)
                ForEach(flagged, id: \.key) { _, room, row in
                    HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
                        Text("Room \(room.id + 1)")
                            .font(.system(size: 13 * u, weight: .bold))
                        Text(row.check.label)
                            .font(.system(size: 13 * u, weight: .semibold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                        Spacer(minLength: 4 * u)
                        Text(row.answer)
                            .font(.system(size: 13 * u, weight: .bold))
                        Text(String(format: "%.2f", row.p))
                            .font(.system(size: 13 * u, weight: .semibold).monospacedDigit())
                    }
                    .foregroundStyle(RoomStyle.flag(.check))
                    .padding(.vertical, 5 * u)
                    .padding(.horizontal, 8 * u)
                    .background(RoomStyle.lane, in: RoundedRectangle(cornerRadius: 8 * u))
                    .contentShape(Rectangle())
                    .onTapGesture { model.selected = room.id }
                }
            }
        }
    }
}
