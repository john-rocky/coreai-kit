// LookView — one screen, dark, top to bottom: the state, the photo, the example photos with Pick
// a photo and Take a photo, the question box with example questions, Ask and Stop, the answer in
// large type, the numbers line, and the model and device it ran on.
//
// Everything sits in one band, the screen's width × 16/9, from the top of the safe area: the same
// layout whether or not a recording crops it (Autoplay writes the band's pixels into the run record
// as `take_rect_px`). The rows have fixed heights (`Rows`); the answer box gets what is left. On an
// iPhone 18 Pro the band is 402 × 714 pt and the box ≈ 234 pt: the answer to "What does this sign
// say?" about the sign example (two paragraphs, 6 lines at 24 pt, ≈ 202 pt) shows whole, from its
// first line to its last.

import PhotosUI
import SwiftUI
#if os(iOS)
import UIKit
#endif

/// The screen's rows, in points (the answer box takes the rest of the band). With the gaps between
/// them, everything but the answer box takes ≈ 480 pt of an iPhone 18 Pro's 714 pt band.
enum Rows {
    static let edge: CGFloat = 4  // above the state, below the footer
    static let status: CGFloat = 28
    static let photoShare: CGFloat = 0.20  // of the band; the photo is fitted inside, its aspect kept
    static let sources: CGFloat = 52
    static let field: CGFloat = 40
    static let chips: CGFloat = 32
    static let ask: CGFloat = 44
    static let answerFont: CGFloat = 24
    static let answerPadding: CGFloat = 10
    static let paragraphGap: CGFloat = 10
    static let numbersFont: CGFloat = 22
    static let footerFont: CGFloat = 11
}

struct LookView: View {
    @State private var model = LookModel()
    @State private var autoplay = Autoplay()
    @State private var pickerItem: PhotosPickerItem?
    @State private var showCamera = false
    @State private var answerScroll = ScrollPosition(edge: .top)
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        GeometryReader { geo in
            let band = min(geo.size.height, (geo.size.width * 16 / 9).rounded(.down))
            VStack(spacing: 0) {
                screen(band: band)
                    .frame(width: geo.size.width, height: band)
                    .onGeometryChange(for: CGRect.self) { $0.frame(in: .global) } action: { model.bandRect = $0 }
                Spacer(minLength: 0)
            }
        }
        .background(Palette.background.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            pickerItem = nil
            Task { await model.pick(item) }
        }
        .onChange(of: scenePhase, initial: true) { _, phase in autoplay.scene(phase) }
        #if os(iOS)
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { model.pick(cameraImage: $0) }.ignoresSafeArea()
        }
        #endif
        .task { await model.load() }
        .task { await autoplay.run(model) }
        #if os(macOS)
        .frame(minWidth: 420, minHeight: 760)
        #endif
    }

    private func screen(band: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            statusRow
            photo(height: (band * Rows.photoShare).rounded()).padding(.top, 6)
            sourceRow.padding(.top, 8)
            questionField.padding(.top, 8)
            chipRow.padding(.top, 6)
            askRow.padding(.top, 8)
            answerBox.padding(.top, 8)
            numbersLine.padding(.top, 6)
            footer.padding(.top, 4)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, Rows.edge)
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
                .accessibilityIdentifier("state.pill")
            Text(model.detail)
                .font(.system(size: 16))
                .foregroundStyle(Palette.secondary)
                .lineLimit(1)
        }
        .frame(height: Rows.status)
    }

    // MARK: - ② photo

    private func photo(height: CGFloat) -> some View {
        ZStack {
            if let photo = model.photo {
                Image(decorative: photo.display, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                RoundedRectangle(cornerRadius: 12).fill(Palette.surface)
                Text("No photo yet").font(.system(size: 17)).foregroundStyle(Palette.secondary)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: height)
    }

    // MARK: - ③ examples, Pick a photo, Take a photo

    private var sourceRow: some View {
        HStack(spacing: 8) {
            ForEach(LookModel.examples) { example in
                Button { model.pickExample(example.id) } label: { thumbnail(example) }
                    .buttonStyle(.plain)
                    .disabled(model.phase == .thinking)
                    .accessibilityLabel("Example: \(example.id)")
                    .accessibilityIdentifier("example.\(example.id)")
            }
            VStack(spacing: 6) {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    sourceButton("Pick a photo")
                }
                .buttonStyle(.plain)
                #if os(iOS)
                Button { showCamera = true } label: { sourceButton("Take a photo") }
                    .buttonStyle(.plain)
                    .disabled(!CameraPicker.isAvailable)
                #endif
            }
            .frame(width: 124)
            .disabled(model.phase == .thinking)
        }
        .frame(height: Rows.sources)
    }

    private func thumbnail(_ example: LookModel.Example) -> some View {
        let selected = model.photo?.source == "example \(example.id)"
        return RoundedRectangle(cornerRadius: 8)
            .fill(Palette.surface)
            .overlay {
                if let thumb = model.thumbnails[example.id] {
                    Image(decorative: thumb, scale: 1).resizable().scaledToFill()
                } else {
                    Text(example.id).font(.system(size: 12)).foregroundStyle(Palette.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: Rows.sources)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(selected ? Palette.accent : .clear, lineWidth: 3))
            .contentShape(Rectangle())
    }

    // nonisolated: PhotosPicker builds its label outside the main actor.
    nonisolated private func sourceButton(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 15, weight: .semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: (Rows.sources - 6) / 2)
            .background(RoundedRectangle(cornerRadius: 8).fill(Palette.surfaceHigh))
    }

    // MARK: - ④ question

    private var questionField: some View {
        TextField("Ask about the photo", text: $model.question)
            .font(.system(size: 20))
            .textFieldStyle(.plain)
            .padding(.horizontal, 14)
            .frame(height: Rows.field)
            .background(RoundedRectangle(cornerRadius: 12).fill(Palette.surface))
            .submitLabel(.go)
            .onSubmit { model.ask() }
            .disabled(model.phase == .thinking)
            .accessibilityIdentifier("question.field")
    }

    private var chipRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(LookModel.chips.indices, id: \.self) { index in
                    Button { model.applyChip(index) } label: {
                        Text(LookModel.chips[index])
                            .font(.system(size: 15))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .frame(height: Rows.chips)
                            .background(Capsule().fill(Palette.surfaceHigh))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("chip.\(index)")
                }
            }
        }
        .frame(height: Rows.chips)
        .disabled(model.phase == .thinking)
    }

    // MARK: - ⑤ Ask, Stop

    private var askRow: some View {
        HStack(spacing: 10) {
            Button { model.ask() } label: {
                Text("Ask")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(model.canAsk ? .white : Palette.secondary)
                    .frame(maxWidth: .infinity)
                    .frame(height: Rows.ask)
                    .background(
                        RoundedRectangle(cornerRadius: 12).fill(model.canAsk ? Palette.accent : Palette.surfaceHigh))
            }
            .buttonStyle(.plain)
            .disabled(!model.canAsk)
            .accessibilityIdentifier("ask")
            Button { model.stop() } label: {
                Text("Stop")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(model.phase == .thinking ? .white : Palette.secondary)
                    .frame(width: 110, height: Rows.ask)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Palette.surfaceHigh))
            }
            .buttonStyle(.plain)
            .disabled(model.phase != .thinking)
            .accessibilityIdentifier("stop")
        }
    }

    // MARK: - ⑥ answer

    /// The answer in large type, paragraph by paragraph (AnswerText.swift). An answer that fits
    /// shows from its first line; one taller than the box follows its last line as it streams, and
    /// scrolls inside the box.
    private var answerBox: some View {
        ScrollView {
            Group {
                if model.answer.isEmpty {
                    Text(model.phase == .thinking ? "…" : "The answer appears here")
                        .foregroundStyle(Palette.secondary)
                } else {
                    VStack(alignment: .leading, spacing: Rows.paragraphGap) {
                        ForEach(Array(model.shownParagraphs.enumerated()), id: \.offset) { item in
                            Text(item.element)
                        }
                    }
                    .foregroundStyle(.white)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(AnswerText.plain(model.answer))
                    .accessibilityIdentifier("answer.text")
                }
            }
            .font(.system(size: Rows.answerFont))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(Rows.answerPadding)
        }
        .scrollPosition($answerScroll)
        .onScrollGeometryChange(for: LookModel.AnswerBox.self) { geometry in
            LookModel.AnswerBox(
                content: geometry.contentSize.height, container: geometry.containerSize.height,
                offset: geometry.contentOffset.y + geometry.contentInsets.top)
        } action: { old, box in
            model.answerBox = box
            // An answer taller than the box: keep its newest line in view while it grows.
            if box.content > box.container + 0.5, box.content > old.content + 0.5 {
                answerScroll.scrollTo(edge: .bottom)
            }
        }
        .onChange(of: model.answer.isEmpty) { _, empty in
            if empty { answerScroll.scrollTo(edge: .top) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 14).fill(Palette.surface))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - ⑦ numbers, ⑧ model and device

    /// Two lines: the measured numbers, then where the model ran.
    private var numbersLine: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(model.numbersHead).lineLimit(1).minimumScaleFactor(0.8)
            Text(model.numbersTail).lineLimit(1).minimumScaleFactor(0.8)
        }
        .font(.system(size: Rows.numbersFont, weight: .bold))
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(model.numbersLine)
        .accessibilityIdentifier("stats.line")
    }

    private var footer: some View {
        Text(model.footer)
            .font(.system(size: Rows.footerFont))
            .foregroundStyle(Palette.secondary)
            .lineLimit(2, reservesSpace: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

enum Palette {
    static let background = Color(red: 0.07, green: 0.08, blue: 0.10)
    static let surface = Color(red: 0.12, green: 0.14, blue: 0.17)
    static let surfaceHigh = Color(red: 0.19, green: 0.21, blue: 0.25)
    static let secondary = Color(white: 0.64)
    static let accent = Color(red: 0.18, green: 0.44, blue: 0.93)
}

extension LookModel.Phase {
    var label: String {
        switch self {
        case .loading: "LOADING"
        case .ready: "READY"
        case .thinking: "THINKING"
        case .done: "DONE"
        case .stopped: "STOPPED"
        case .failed: "FAILED"
        }
    }

    var color: Color {
        switch self {
        case .loading, .stopped: Color(white: 0.36)
        case .ready: Palette.accent
        case .thinking: Color(red: 0.84, green: 0.50, blue: 0.08)
        case .done: Color(red: 0.16, green: 0.50, blue: 0.22)
        case .failed: Color(red: 0.78, green: 0.20, blue: 0.20)
        }
    }
}

#if os(iOS)
/// The system camera, for one photo. It asks for the camera permission the first time
/// (NSCameraUsageDescription); Pick a photo needs none.
struct CameraPicker: UIViewControllerRepresentable {
    let onPhoto: @MainActor (UIImage) -> Void
    @Environment(\.dismiss) var dismiss

    static var isAvailable: Bool { UIImagePickerController.isSourceTypeAvailable(.camera) }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ picker: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPhoto: onPhoto, close: { dismiss() })
    }

    @MainActor
    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onPhoto: @MainActor (UIImage) -> Void
        let close: @MainActor () -> Void

        init(onPhoto: @escaping @MainActor (UIImage) -> Void, close: @escaping @MainActor () -> Void) {
            self.onPhoto = onPhoto
            self.close = close
        }

        func imagePickerController(
            _ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let image = info[.originalImage] as? UIImage { onPhoto(image) }
            close()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            close()
        }
    }
}
#endif
