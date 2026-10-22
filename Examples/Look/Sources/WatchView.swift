// WatchView — the watch screen, dark, top to bottom: the state, the source (Camera, Pick a video,
// Example photos), the picture with the reaction over it, the rules (a tap opens the editor; "+ Add
// a rule" under them), Start / Stop, the numbers lines, and the model and device it ran on.
//
// Everything sits in the band the app gives every screen (the screen's width × 16/9, LookApp.swift).
// The rows have fixed heights and the picture gets what is left: on an iPhone 18 Pro with three rules
// it is ≈ 275 pt tall in take mode (`-take 1`: three rules at most, no "+ Add a rule", no video
// button) and ≈ 229 pt otherwise. A rule's words are 26 pt (a long sentence shrinks to fit its row);
// a reaction shows its symbol at 64 pt and its words at 44 pt for at least 3 s; an effect plays over
// its place on the picture for 3 s, drawn (WatchEffects.swift), without a symbol or words.

import PhotosUI
import SwiftUI

/// The watch screen's rows, in points (the picture takes the rest of the band).
enum WatchRows {
    static let edge: CGFloat = 4
    static let status: CGFloat = 28
    static let sources: CGFloat = 36
    static let row: CGFloat = 64
    static let rowGap: CGFloat = 6
    static let add: CGFloat = 40
    static let logLine: CGFloat = 19
    static let start: CGFloat = 50
    static let numbers: CGFloat = 46
    static let footer: CGFloat = 27
    static let conditionFont: CGFloat = 26
    static let overlaySymbol: CGFloat = 64
    static let overlayFont: CGFloat = 44
    /// Rules shown before the list scrolls (and all a take shows).
    static let shown = 3
}

enum WatchPalette {
    static let yes = Color(red: 0.16, green: 0.55, blue: 0.24)
    static let yesRow = Color(red: 0.10, green: 0.27, blue: 0.15)
    static let unsure = Color(red: 0.84, green: 0.50, blue: 0.08)
    static let stop = Color(red: 0.62, green: 0.18, blue: 0.18)
    static let speech = Color(red: 0.18, green: 0.44, blue: 0.93).opacity(0.92)
}

struct WatchView: View {
    @Bindable var model: WatchModel
    let band: CGFloat
    let take: Bool
    @State private var videoItem: PhotosPickerItem?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            statusRow
            sourceRow.padding(.top, 6)
            picture(height: pictureHeight).padding(.top, 8)
            ruleList.padding(.top, 8)
            if model.showsLog { logBox.padding(.top, 6) }
            startButton.padding(.top, 8)
            numbers.padding(.top, 6)
            footer.padding(.top, 4)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, WatchRows.edge)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .onChange(of: videoItem) { _, item in
            guard let item else { return }
            videoItem = nil
            Task { await model.openVideo(item) }
        }
        .sheet(item: $model.editing) { target in
            RuleEditor(
                target: target, count: model.rowStates[target.rule.id]?.count ?? 0,
                onSave: { model.save($0) }, onDelete: { model.delete(target.rule.id) },
                onPlay: { model.preview($0) }, onPlayEffect: { model.preview($0) })
        }
    }

    // MARK: - Heights

    private var shownRules: [WatchRule] { take ? Array(model.rules.prefix(WatchRows.shown)) : model.rules }

    private var showsAdd: Bool { !take && model.rules.count < WatchRules.maxCount }

    private var listHeight: CGFloat {
        let rows = min(model.rules.count, WatchRows.shown)
        var height = CGFloat(rows) * WatchRows.row + CGFloat(max(rows - 1, 0)) * WatchRows.rowGap
        if showsAdd { height += (rows > 0 ? WatchRows.rowGap : 0) + WatchRows.add }
        return height
    }

    private var logHeight: CGFloat { model.showsLog ? 6 + 3 * WatchRows.logLine : 0 }

    private var pictureHeight: CGFloat {
        let fixed =
            2 * WatchRows.edge + WatchRows.status + 6 + WatchRows.sources + 8 + 8 + listHeight + logHeight + 8
            + WatchRows.start + 6 + WatchRows.numbers + 4 + WatchRows.footer
        return max(140, (band - fixed).rounded(.down))
    }

    // MARK: - ① state

    private var statusRow: some View {
        HStack(spacing: 10) {
            Text(model.phase.label)
                .font(.system(size: 15, weight: .heavy))
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
                .background(Capsule().fill(model.phase.color))
                .accessibilityIdentifier("watch.pill")
            Text(model.detail)
                .font(.system(size: 16))
                .foregroundStyle(Palette.secondary)
                .lineLimit(1)
        }
        .frame(height: WatchRows.status)
    }

    // MARK: - ② source

    private var sourceRow: some View {
        let source = model.source  // read here: PhotosPicker builds its label outside the main actor
        return HStack(spacing: 8) {
            Button { model.select(.camera) } label: {
                sourceLabel("Camera", selected: source == .camera)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("watch.source.camera")
            PhotosPicker(selection: $videoItem, matching: .videos) {
                sourceLabel("Pick a video", selected: source == .video)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("watch.source.video")
            Button { model.select(.photos) } label: {
                sourceLabel("Example photos", selected: source == .photos)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("watch.source.photos")
        }
        .frame(height: WatchRows.sources)
        .disabled(model.phase == .watching || model.phase == .loading)
    }

    // nonisolated: PhotosPicker builds its label outside the main actor.
    nonisolated private func sourceLabel(_ title: String, selected: Bool) -> some View {
        Text(title)
            .font(.system(size: 15, weight: .semibold))
            .foregroundStyle(.white)
            .lineLimit(1)
            .minimumScaleFactor(0.8)
            .frame(maxWidth: .infinity)
            .frame(height: WatchRows.sources)
            .background(RoundedRectangle(cornerRadius: 8).fill(selected ? Palette.accent : Palette.surfaceHigh))
    }

    // MARK: - ③ picture and reaction

    private func picture(height: CGFloat) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12).fill(Palette.surface)
            switch model.source {
            case .camera:
                #if os(iOS)
                if let camera = model.camera {
                    CameraPreview(session: camera.session)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    placeholder("camera")
                }
                #else
                placeholder("camera")
                #endif
            case .video:
                if let video = model.video {
                    PlayerPreview(player: video.player)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    placeholder("film")
                }
            case .photos:
                if let image = model.shownImage {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    placeholder("photo")
                }
            }
            WatchEffectLayer(runs: model.effects)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            if let overlay = model.overlay {
                reaction(overlay)
                    .id(overlay.id)
                    .transition(.opacity)
            }
            if model.source == .video, let video = model.video, !take {
                VideoButton(video: video)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                    .padding(10)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: height)
        .onGeometryChange(for: CGSize.self) { $0.size } action: { model.previewSize = $0 }
    }

    private func placeholder(_ symbol: String) -> some View {
        Image(systemName: symbol)
            .font(.system(size: 44))
            .foregroundStyle(Palette.secondary)
    }

    private func reaction(_ overlay: WatchModel.Overlay) -> some View {
        VStack(spacing: 6) {
            HStack(spacing: 16) {
                ForEach(Array(overlay.symbols.enumerated()), id: \.offset) { item in
                    Image(systemName: item.element)
                        .font(.system(size: WatchRows.overlaySymbol, weight: .semibold))
                }
            }
            Text(overlay.title)
                .font(.system(size: WatchRows.overlayFont, weight: .bold))
                .lineLimit(2)
                .minimumScaleFactor(0.8)
                .multilineTextAlignment(.center)
            if !overlay.subtitle.isEmpty {
                Text(overlay.subtitle)
                    .font(.system(size: 24, weight: .semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .foregroundStyle(.white.opacity(0.85))
            }
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 22).fill(overlay.speech ? WatchPalette.speech : Color.black.opacity(0.62))
        )
        .padding(6)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("watch.reaction")
    }

    // MARK: - ④ rules

    private var ruleList: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                VStack(spacing: WatchRows.rowGap) {
                    ForEach(Array(shownRules.enumerated()), id: \.element.id) { item in
                        ruleRow(item.element, number: item.offset + 1)
                    }
                    if showsAdd {
                        Button { model.addRule() } label: {
                            Label("Add a rule", systemImage: "plus")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundStyle(.white)
                                .frame(maxWidth: .infinity)
                                .frame(height: WatchRows.add)
                                .background(RoundedRectangle(cornerRadius: 12).stroke(Palette.surfaceHigh, lineWidth: 2))
                        }
                        .buttonStyle(.plain)
                        .disabled(model.phase == .watching)
                        .accessibilityIdentifier("watch.add")
                    }
                }
            }
            .scrollBounceBehavior(.basedOnSize)
            .onChange(of: model.rules.count) { before, after in
                // A rule just added shows at the bottom of the list.
                guard after > before, let last = model.rules.last else { return }
                withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
        .frame(height: listHeight)
    }

    private func ruleRow(_ rule: WatchRule, number: Int) -> some View {
        let state = model.rowStates[rule.id] ?? WatchModel.RowState()
        let yes = state.answer == "yes"
        return Button { model.edit(rule) } label: {
            HStack(spacing: 12) {
                Image(systemName: rule.symbol)
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 44, height: 44)
                    .background(RoundedRectangle(cornerRadius: 10).fill(yes ? WatchPalette.yes : Palette.surfaceHigh))
                VStack(alignment: .leading, spacing: 1) {
                    // a sentence ("water rushing over rows of stones") shrinks to fit the row rather than lose its end
                    Text(rule.condition)
                        .font(.system(size: WatchRows.conditionFont, weight: .semibold))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .minimumScaleFactor(0.6)
                    HStack(spacing: 8) {
                        Text(Self.answerLabel(state.answer))
                            .font(.system(size: 14, weight: .heavy))
                            .foregroundStyle(.white)
                            .frame(minWidth: 46)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Self.answerColor(state.answer)))
                        reactionMark(rule, state)
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(Palette.secondary)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .frame(height: WatchRows.row)
            .background(RoundedRectangle(cornerRadius: 12).fill(yes ? WatchPalette.yesRow : Palette.surface))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // Not `.disabled`: a disabled Button dims its label, and the rows show the answers while a
        // run goes on. edit(_:) ignores a tap while watching.
        .allowsHitTesting(model.phase != .watching)
        .accessibilityIdentifier("watch.rule.\(number)")
    }

    /// What the rule does, on its row: "chime", "say", "3:00" (the time left while it runs), "×2", …
    @ViewBuilder
    private func reactionMark(_ rule: WatchRule, _ state: WatchModel.RowState) -> some View {
        switch rule.reaction {
        case .sound(let sound):
            Label(sound.rawValue, systemImage: "speaker.wave.2")
        case .haptic:
            Label("buzz", systemImage: "iphone.radiowaves.left.and.right")
        case .say:
            Label("say", systemImage: "text.bubble")
        case .timer(let minutes):
            if let end = state.timerEnd {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    Label(WatchReaction.clock(end.timeIntervalSince(context.date)), systemImage: "timer")
                }
            } else {
                Label(WatchReaction.clock(minutes * 60), systemImage: "timer")
            }
        case .count:
            Label("×\(state.count)", systemImage: "number")
        case .log:
            Label("log", systemImage: "list.bullet")
        case .effect(let effect):
            Label(effect.rawValue, systemImage: "sparkles")
        }
    }

    private static func answerLabel(_ answer: String) -> String {
        switch answer {
        case "yes": "YES"
        case "no": "NO"
        case "?": "?"
        default: "–"
        }
    }

    private static func answerColor(_ answer: String) -> Color {
        switch answer {
        case "yes": WatchPalette.yes
        case "?": WatchPalette.unsure
        default: Palette.surfaceHigh
        }
    }

    // MARK: - ⑤ log

    private var logBox: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(0..<3, id: \.self) { index in
                Text(index < model.logLines.count ? model.logLines[index] : (index == 0 ? "The log is empty" : " "))
                    .font(.system(size: 15, design: .monospaced))
                    .foregroundStyle(index < model.logLines.count ? .white : Palette.secondary)
                    .lineLimit(1)
                    .frame(height: WatchRows.logLine, alignment: .leading)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityIdentifier("watch.log")
    }

    // MARK: - ⑥ Start / Stop

    private var startButton: some View {
        let watching = model.phase == .watching
        let enabled = watching ? !model.stopping : model.canStart
        return Button {
            if watching { model.stop() } else { model.start() }
        } label: {
            Text(watching ? (model.stopping ? "Stopping…" : "Stop") : "Start")
                .font(.system(size: 24, weight: .bold))
                .foregroundStyle(enabled ? .white : Palette.secondary)
                .frame(maxWidth: .infinity)
                .frame(height: WatchRows.start)
                .background(
                    RoundedRectangle(cornerRadius: 12).fill(
                        watching ? WatchPalette.stop : (enabled ? Palette.accent : Palette.surfaceHigh)))
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .accessibilityIdentifier("watch.start")
    }

    // MARK: - ⑦ numbers, ⑧ model and device

    private var numbers: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(model.numbersHead)
                .font(.system(size: 22, weight: .bold))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Text(model.numbersTail)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Palette.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: WatchRows.numbers)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(model.numbersLine)
        .accessibilityIdentifier("watch.numbers")
    }

    private var footer: some View {
        Text(model.footer)
            .font(.system(size: 11))
            .foregroundStyle(Palette.secondary)
            .lineLimit(2, reservesSpace: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(height: WatchRows.footer, alignment: .top)
    }
}

/// Play / pause for a picked video, over the picture's corner (not in take mode).
private struct VideoButton: View {
    let video: VideoFeed
    @State private var playing = true

    var body: some View {
        Button {
            if playing { video.pause() } else { video.play() }
            playing.toggle()
        } label: {
            Image(systemName: playing ? "pause.fill" : "play.fill")
                .font(.system(size: 18, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .background(Circle().fill(Color.black.opacity(0.55)))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(playing ? "Pause the video" : "Play the video")
        .onAppear { playing = video.isPlaying }
    }
}

/// One rule in a sheet: what to look for, what happens then, its symbol (not for an effect, which
/// shows none); Done keeps it.
struct RuleEditor: View {
    let target: WatchModel.EditTarget
    let count: Int
    let onSave: (WatchRule) -> Void
    let onDelete: () -> Void
    let onPlay: (WatchSound) -> Void
    let onPlayEffect: (WatchEffect) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var condition: String
    @State private var kind: WatchReaction.Kind
    @State private var sound: WatchSound
    @State private var sayText: String
    @State private var minutes: Double
    @State private var effect: WatchEffect
    @State private var symbol: String

    init(
        target: WatchModel.EditTarget, count: Int, onSave: @escaping (WatchRule) -> Void,
        onDelete: @escaping () -> Void, onPlay: @escaping (WatchSound) -> Void,
        onPlayEffect: @escaping (WatchEffect) -> Void
    ) {
        self.target = target
        self.count = count
        self.onSave = onSave
        self.onDelete = onDelete
        self.onPlay = onPlay
        self.onPlayEffect = onPlayEffect
        let rule = target.rule
        var sound = WatchSound.chime
        var say = ""
        var minutes = 3.0
        var effect = WatchEffect.ripple
        switch rule.reaction {
        case .sound(let chosen): sound = chosen
        case .say(let text): say = text
        case .timer(let chosen): minutes = chosen
        case .effect(let chosen): effect = chosen
        default: break
        }
        _condition = State(initialValue: rule.condition)
        _kind = State(initialValue: rule.reaction.kind)
        _sound = State(initialValue: sound)
        _sayText = State(initialValue: say)
        _minutes = State(initialValue: minutes)
        _effect = State(initialValue: effect)
        _symbol = State(initialValue: rule.symbol)
    }

    private var trimmed: String { condition.trimmingCharacters(in: .whitespacesAndNewlines) }

    private var reaction: WatchReaction {
        switch kind {
        case .sound: .sound(sound)
        case .haptic: .haptic
        case .say: .say(sayText.trimmingCharacters(in: .whitespacesAndNewlines))
        case .timer: .timer(minutes: min(max(minutes, 0.1), 120))
        case .count: .count
        case .log: .log
        case .effect: .effect(effect)
        }
    }

    private var canSave: Bool {
        !trimmed.isEmpty && (kind != .say || !sayText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            && (kind != .timer || minutes > 0)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("a coffee mug", text: $condition)
                        .font(.system(size: 22))
                        .autocorrectionDisabled()
                        #if os(iOS)
                        .textInputAutocapitalization(.never)
                        #endif
                        .accessibilityIdentifier("editor.condition")
                } header: {
                    Text("Look for")
                } footer: {
                    Text("Every frame, the model is asked: Is there \(trimmed.isEmpty ? "…" : trimmed)?")
                }
                Section("Then") {
                    Picker("Reaction", selection: $kind) {
                        ForEach(WatchReaction.Kind.allCases) { kind in
                            Label(kind.title, systemImage: kind.icon).tag(kind)
                        }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                    detail
                }
                if kind != .effect {
                    Section("Symbol") {
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 6), spacing: 8) {
                            ForEach(WatchRules.symbols, id: \.self) { name in
                                Button { symbol = name } label: {
                                    Image(systemName: name)
                                        .font(.system(size: 22))
                                        .foregroundStyle(.white)
                                        .frame(maxWidth: .infinity)
                                        .frame(height: 44)
                                        .background(
                                            RoundedRectangle(cornerRadius: 8).fill(symbol == name ? Palette.accent : Palette.surfaceHigh))
                                }
                                .buttonStyle(.plain)
                                .accessibilityLabel(name)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
                if !target.isNew {
                    Section {
                        Button("Delete this rule", role: .destructive) { onDelete() }
                    }
                }
            }
            .navigationTitle(target.isNew ? "New rule" : "Rule")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        var rule = target.rule
                        rule.condition = trimmed
                        rule.reaction = reaction
                        rule.symbol = symbol
                        onSave(rule)
                    }
                    .disabled(!canSave)
                    .accessibilityIdentifier("editor.done")
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    @ViewBuilder
    private var detail: some View {
        switch kind {
        case .sound:
            Picker("Sound", selection: $sound) {
                ForEach(WatchSound.allCases) { sound in Text(sound.title).tag(sound) }
            }
            .pickerStyle(.segmented)
            Button("Play it") { onPlay(sound) }
        case .haptic:
            Text("A short buzz in your hand.").foregroundStyle(.secondary)
        case .say:
            TextField("There is the lighthouse", text: $sayText)
            Text("Spoken by the system voice, in English.").foregroundStyle(.secondary)
        case .timer:
            HStack {
                TextField("3", value: $minutes, format: .number)
                    #if os(iOS)
                    .keyboardType(.decimalPad)
                    #endif
                    .frame(maxWidth: 90)
                Text("minutes")
            }
            Text("Starts a timer on the rule's row; at its end, a chime.").foregroundStyle(.secondary)
        case .count:
            Text("Adds one each time it fires (now \(count)).").foregroundStyle(.secondary)
        case .log:
            Text("Writes the time and the rule under the list.").foregroundStyle(.secondary)
        case .effect:
            Picker("Effect", selection: $effect) {
                ForEach(WatchEffect.allCases) { effect in Text(effect.title).tag(effect) }
            }
            .pickerStyle(.segmented)
            Text(effect.look).foregroundStyle(.secondary)
            Text("It plays over the part of the picture where the model says the thing is, or over all of it.")
                .foregroundStyle(.secondary)
            Button("Play its sound") { onPlayEffect(effect) }
        }
    }
}
