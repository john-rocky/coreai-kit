// LookView — one screen, dark, top to bottom: the state, the photo, the example photos with Pick
// a photo and Take a photo, the question box with example questions, Ask and Stop, the answer in
// large type, the numbers line, and the model and device it ran on.
//
// `-take 1` lays the whole screen out in a 9:16 band under the status bar — the frame a recording
// crops; Autoplay writes the band's pixels into the run record as `take_rect_px`.

import PhotosUI
import SwiftUI
#if os(iOS)
import UIKit
#endif

struct LookView: View {
    @State private var model = LookModel()
    @State private var autoplay = Autoplay()
    @State private var pickerItem: PhotosPickerItem?
    @State private var showCamera = false
    private let take = UserDefaults.standard.bool(forKey: "take")

    var body: some View {
        GeometryReader { geo in
            let band = take ? min(geo.size.height, (geo.size.width * 16 / 9).rounded(.down)) : geo.size.height
            VStack(spacing: 0) {
                screen(height: band)
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
        #if os(iOS)
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { model.pick(cameraImage: $0) }.ignoresSafeArea()
        }
        #endif
        .task { await model.load() }
        .task { await autoplay.run(model) }
        #if os(macOS)
        .frame(minWidth: 420, minHeight: take ? 760 : 820)
        #endif
    }

    private func screen(height: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            statusRow
            photo(height: (height * (take ? 0.26 : 0.30)).rounded())
            sourceRow
            questionBox
            askRow
            answerBox
            numbersLine
            footer
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    // MARK: - ① state

    private var statusRow: some View {
        HStack(spacing: 10) {
            Text(model.phase.label)
                .font(.system(size: 15, weight: .heavy))
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 5)
                .background(Capsule().fill(model.phase.color))
            Text(model.detail)
                .font(.system(size: 16))
                .foregroundStyle(Palette.secondary)
                .lineLimit(1)
        }
        .frame(height: 30)
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
        .frame(height: 58)
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
            .frame(height: 58)
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
            .frame(height: 26)
            .background(RoundedRectangle(cornerRadius: 8).fill(Palette.surfaceHigh))
    }

    // MARK: - ④ question

    private var questionBox: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextField("Ask about the photo", text: $model.question)
                .font(.system(size: 20))
                .textFieldStyle(.plain)
                .padding(.horizontal, 14)
                .frame(height: 46)
                .background(RoundedRectangle(cornerRadius: 12).fill(Palette.surface))
                .submitLabel(.go)
                .onSubmit { model.ask() }
                .disabled(model.phase == .thinking)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(LookModel.chips.indices, id: \.self) { index in
                        Button { model.applyChip(index) } label: {
                            Text(LookModel.chips[index])
                                .font(.system(size: 15))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 12)
                                .frame(height: 32)
                                .background(Capsule().fill(Palette.surfaceHigh))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .frame(height: 32)
            .disabled(model.phase == .thinking)
        }
    }

    // MARK: - ⑤ Ask, Stop

    private var askRow: some View {
        HStack(spacing: 10) {
            Button { model.ask() } label: {
                Text("Ask")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(model.canAsk ? .white : Palette.secondary)
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(
                        RoundedRectangle(cornerRadius: 12).fill(model.canAsk ? Palette.accent : Palette.surfaceHigh))
            }
            .buttonStyle(.plain)
            .disabled(!model.canAsk)
            Button { model.stop() } label: {
                Text("Stop")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(model.phase == .thinking ? .white : Palette.secondary)
                    .frame(width: 110, height: 50)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Palette.surfaceHigh))
            }
            .buttonStyle(.plain)
            .disabled(model.phase != .thinking)
        }
    }

    // MARK: - ⑥ answer

    /// The box shows the answer from its first line; an answer taller than the box scrolls inside it.
    /// On an iPhone 18 Pro (iOS 27.2) the bottom anchor for size changes did not move the box while a
    /// longer answer streamed in: the first line stayed in view.
    private var answerBox: some View {
        ScrollView {
            Group {
                if model.answer.isEmpty {
                    Text(model.phase == .thinking ? "…" : "The answer appears here")
                        .foregroundStyle(Palette.secondary)
                } else {
                    Text(LookModel.shown(model.answer)).foregroundStyle(.white)
                }
            }
            .font(.system(size: 26))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
        }
        .defaultScrollAnchor(.top, for: .initialOffset)
        .defaultScrollAnchor(.top, for: .alignment)
        .defaultScrollAnchor(.bottom, for: .sizeChanges)
        .onScrollGeometryChange(for: LookModel.AnswerBox.self) { geometry in
            LookModel.AnswerBox(
                content: geometry.contentSize.height, container: geometry.containerSize.height,
                offset: geometry.contentOffset.y + geometry.contentInsets.top)
        } action: { _, box in
            model.answerBox = box
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 14).fill(Palette.surface))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - ⑦ numbers, ⑧ model and device

    /// One line where it fits; on a phone the measured numbers keep the first line to themselves.
    private var numbersLine: some View {
        ViewThatFits(in: .horizontal) {
            Text(model.numbersLine).lineLimit(1)
            VStack(alignment: .leading, spacing: 0) {
                Text(model.numbersHead).lineLimit(1).minimumScaleFactor(0.85)
                Text(model.numbersTail).lineLimit(1).minimumScaleFactor(0.85)
            }
        }
        .font(.system(size: 22, weight: .bold))
        .foregroundStyle(.white)
    }

    private var footer: some View {
        Text(model.footer)
            .font(.system(size: 11))
            .foregroundStyle(Palette.secondary)
            .lineLimit(2)
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
