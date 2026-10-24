// CutoutView — the screen, dark, top to bottom: the state; the photo with the model's boxes, or the cut-out over a
// checkerboard; the example photos with Pick a photo and Take a photo; what to cut out, with words to tap; Cut out /
// Boxes only; Cut out, Save and Stop; the pieces the model found (tap one to leave it out of the PNG); the numbers line;
// the model and the device it ran on.
//
// Everything sits in one band, the screen's width × 16/9 (CutoutApp.swift); the photo gets what the fixed rows leave.

import PhotosUI
import SwiftUI
#if os(iOS)
import UIKit
#endif

enum Rows {
    static let edge: CGFloat = 4
    static let status: CGFloat = 28
    static let sources: CGFloat = 52
    static let field: CGFloat = 40
    static let chips: CGFloat = 32
    static let mode: CGFloat = 32
    static let action: CGFloat = 44
    static let pieces: CGFloat = 34
    static let numbersFont: CGFloat = 20
    static let footerFont: CGFloat = 11
}

struct CutoutView: View {
    @Bindable var model: CutoutModel
    let band: CGFloat
    @State private var pickerItem: PhotosPickerItem?
    @State private var showCamera = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            statusRow
            picture.padding(.top, 6)
            sourceRow.padding(.top, 8)
            wordsField.padding(.top, 8)
            chipRow.padding(.top, 6)
            modeRow.padding(.top, 6)
            actionRow.padding(.top, 8)
            piecesRow.padding(.top, 6)
            numbersLine.padding(.top, 6)
            footer.padding(.top, 4)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, Rows.edge)
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
    }

    // MARK: - state

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
                .lineLimit(2)
                .minimumScaleFactor(0.8)
                .accessibilityIdentifier("state.detail")
        }
        .frame(minHeight: Rows.status)
    }

    // MARK: - the photo / the cut-out

    private var picture: some View {
        GeometryReader { geo in
            ZStack {
                RoundedRectangle(cornerRadius: 12).fill(Palette.surface)
                if let photo = model.photo {
                    let fit = Self.fitted(CGSize(width: photo.rgb.width, height: photo.rgb.height), in: geo.size)
                    if model.showCutout, let cut = model.cutoutImage {
                        Checkerboard().frame(width: fit.width, height: fit.height).position(x: fit.midX, y: fit.midY)
                        Image(decorative: cut, scale: 1).resizable()
                            .frame(width: fit.width, height: fit.height).position(x: fit.midX, y: fit.midY)
                            .accessibilityIdentifier("cutout.image")
                    } else {
                        Image(decorative: photo.display, scale: 1).resizable()
                            .frame(width: fit.width, height: fit.height).position(x: fit.midX, y: fit.midY)
                        ForEach(model.pieces) { piece in
                            boxOverlay(piece, in: fit)
                        }
                    }
                } else {
                    Text("No photo yet").font(.system(size: 17)).foregroundStyle(Palette.secondary)
                }
                if model.phase == .done, model.task == .cutout, model.cutoutImage != nil {
                    VStack {
                        HStack {
                            Spacer()
                            Picker("View", selection: $model.showCutout) {
                                Text("Photo").tag(false)
                                Text("Cut out").tag(true)
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                            .frame(width: 170)
                            .padding(8)
                            .accessibilityIdentifier("view.switch")
                        }
                        Spacer()
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func boxOverlay(_ piece: CutoutModel.Piece, in fit: CGRect) -> some View {
        let x1: CGFloat = CGFloat(piece.box[0]) / 1000, y1: CGFloat = CGFloat(piece.box[1]) / 1000
        let x2: CGFloat = CGFloat(piece.box[2]) / 1000, y2: CGFloat = CGFloat(piece.box[3]) / 1000
        let left: CGFloat = fit.minX + fit.width * x1
        let top: CGFloat = fit.minY + fit.height * y1
        let r = CGRect(x: left, y: top, width: fit.width * (x2 - x1), height: fit.height * (y2 - y1))
        let color = piece.selected ? Palette.box : Palette.secondary
        return ZStack(alignment: .topLeading) {
            Rectangle().stroke(color, lineWidth: 3)
            Text("\(piece.id + 1)")
                .font(.system(size: 13, weight: .bold))
                .foregroundStyle(.black)
                .padding(.horizontal, 5)
                .background(color)
        }
        .frame(width: max(r.width, 1), height: max(r.height, 1))
        .position(x: r.midX, y: r.midY)
    }

    static func fitted(_ size: CGSize, in box: CGSize) -> CGRect {
        let s = min(box.width / max(size.width, 1), box.height / max(size.height, 1))
        let w = size.width * s, h = size.height * s
        return CGRect(x: (box.width - w) / 2, y: (box.height - h) / 2, width: w, height: h)
    }

    // MARK: - examples, Pick a photo, Take a photo

    private var sourceRow: some View {
        HStack(spacing: 8) {
            ForEach(CutoutModel.examples) { example in
                Button { model.pickExample(example.id) } label: { thumbnail(example) }
                    .buttonStyle(.plain)
                    .disabled(model.phase == .working)
                    .accessibilityLabel("Example: \(example.id)")
                    .accessibilityIdentifier("example.\(example.id)")
            }
            VStack(spacing: 6) {
                PhotosPicker(selection: $pickerItem, matching: .images) { sourceButton("Pick a photo") }
                    .buttonStyle(.plain)
                #if os(iOS)
                Button { showCamera = true } label: { sourceButton("Take a photo") }
                    .buttonStyle(.plain)
                    .disabled(!CameraPicker.isAvailable)
                #endif
            }
            .frame(width: 124)
            .disabled(model.phase == .working)
        }
        .frame(height: Rows.sources)
    }

    private func thumbnail(_ example: CutoutModel.Example) -> some View {
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

    nonisolated private func sourceButton(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 15, weight: .semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: (Rows.sources - 6) / 2)
            .background(RoundedRectangle(cornerRadius: 8).fill(Palette.surfaceHigh))
    }

    // MARK: - what to cut out

    private var wordsField: some View {
        TextField("What to cut out (\"the dog\")", text: $model.words)
            .font(.system(size: 20))
            .textFieldStyle(.plain)
            .padding(.horizontal, 14)
            .frame(height: Rows.field)
            .background(RoundedRectangle(cornerRadius: 12).fill(Palette.surface))
            .submitLabel(.go)
            .onSubmit { model.cutOut() }
            .disabled(model.phase == .working)
            .accessibilityIdentifier("words.field")
    }

    private var chipRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(CutoutModel.chips.indices, id: \.self) { index in
                    Button { model.setWords(CutoutModel.chips[index]) } label: {
                        Text(CutoutModel.chips[index])
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
        .disabled(model.phase == .working)
    }

    private var modeRow: some View {
        Picker("Task", selection: $model.task) {
            Text("Cut out").tag(CutoutTask.cutout)
            Text("Boxes only").tag(CutoutTask.boxes)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(height: Rows.mode)
        .disabled(model.phase == .working)
        .accessibilityIdentifier("task.switch")
    }

    // MARK: - Cut out, Save, Stop

    private var actionRow: some View {
        HStack(spacing: 10) {
            Button { model.cutOut() } label: {
                Text(model.task == .cutout ? "Cut out" : "Find")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(model.canRun ? .white : Palette.secondary)
                    .frame(maxWidth: .infinity)
                    .frame(height: Rows.action)
                    .background(RoundedRectangle(cornerRadius: 12).fill(model.canRun ? Palette.accent : Palette.surfaceHigh))
            }
            .buttonStyle(.plain)
            .disabled(!model.canRun)
            .accessibilityIdentifier("run")
            let canSave = model.phase == .done && model.task == .cutout && model.pieces.contains(where: \.selected)
            Button { model.save() } label: {
                Text(saveTitle)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(canSave ? .white : Palette.secondary)
                    .frame(width: 120, height: Rows.action)
                    .background(RoundedRectangle(cornerRadius: 12).fill(canSave ? Palette.save : Palette.surfaceHigh))
            }
            .buttonStyle(.plain)
            .disabled(!canSave)
            .accessibilityIdentifier("save")
            Button { model.stop() } label: {
                Text("Stop")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(model.phase == .working ? .white : Palette.secondary)
                    .frame(width: 72, height: Rows.action)
                    .background(RoundedRectangle(cornerRadius: 12).fill(Palette.surfaceHigh))
            }
            .buttonStyle(.plain)
            .disabled(model.phase != .working)
            .accessibilityIdentifier("stop")
        }
    }

    private var saveTitle: String {
        #if os(iOS)
        "Save to Photos"
        #else
        "Save PNG…"
        #endif
    }

    // MARK: - the pieces

    private var piecesRow: some View {
        Group {
            if let line = model.saveLine {
                Text(line)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("save.line")
            } else if model.pieces.isEmpty {
                Text(model.answer.isEmpty ? " " : "answer: \(model.answer)")
                    .font(.system(size: 14, design: .monospaced))
                    .foregroundStyle(Palette.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("answer.line")
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        // one label for all the pieces: numbers only (the label is in the state line)
                        let uniform = Set(model.pieces.map(\.label)).count == 1
                        ForEach(model.pieces) { piece in
                            Button { model.toggle(piece.id) } label: {
                                Text(uniform ? "  \(piece.id + 1)  " : "\(piece.id + 1) \(piece.label)")
                                    .font(.system(size: 15, weight: .semibold))
                                    .foregroundStyle(piece.selected ? .black : .white)
                                    .lineLimit(1)
                                    .padding(.horizontal, 10)
                                    .frame(height: Rows.pieces - 4)
                                    .background(Capsule().fill(piece.selected ? Palette.box : Palette.surfaceHigh))
                            }
                            .buttonStyle(.plain)
                            .disabled(model.task != .cutout)
                            .accessibilityIdentifier("piece.\(piece.id)")
                        }
                    }
                }
            }
        }
        .frame(height: Rows.pieces)
    }

    // MARK: - numbers, model and device

    private var numbersLine: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(model.numbersHead).lineLimit(1).minimumScaleFactor(0.7)
            Text(model.numbersTail).lineLimit(1).minimumScaleFactor(0.8)
        }
        .font(.system(size: Rows.numbersFont, weight: .bold))
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(model.numbersHead) · \(model.numbersTail)")
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

/// Grey and white squares behind the cut-out: where it is transparent.
struct Checkerboard: View {
    var body: some View {
        Canvas { ctx, size in
            let s: CGFloat = 12
            for y in stride(from: 0, to: size.height, by: s) {
                for x in stride(from: 0, to: size.width, by: s) {
                    let dark = (Int(x / s) + Int(y / s)) % 2 == 0
                    ctx.fill(Path(CGRect(x: x, y: y, width: s, height: s)), with: .color(dark ? Color(white: 0.55) : Color(white: 0.75)))
                }
            }
        }
    }
}

enum Palette {
    static let background = Color(red: 0.07, green: 0.08, blue: 0.10)
    static let surface = Color(red: 0.12, green: 0.14, blue: 0.17)
    static let surfaceHigh = Color(red: 0.19, green: 0.21, blue: 0.25)
    static let secondary = Color(white: 0.64)
    static let accent = Color(red: 0.18, green: 0.44, blue: 0.93)
    static let save = Color(red: 0.16, green: 0.50, blue: 0.22)
    static let box = Color(red: 1.0, green: 0.80, blue: 0.10)
}

extension CutoutModel.Phase {
    var label: String {
        switch self {
        case .loading: "LOADING"
        case .ready: "READY"
        case .working: "WORKING"
        case .done: "DONE"
        case .nothing: "NONE"
        case .stopped: "STOPPED"
        case .failed: "FAILED"
        }
    }

    var color: Color {
        switch self {
        case .loading, .stopped: Color(white: 0.36)
        case .ready: Palette.accent
        case .working: Color(red: 0.84, green: 0.50, blue: 0.08)
        case .done: Color(red: 0.16, green: 0.50, blue: 0.22)
        case .nothing: Color(red: 0.55, green: 0.40, blue: 0.10)
        case .failed: Color(red: 0.78, green: 0.20, blue: 0.20)
        }
    }
}

#if os(iOS)
/// The system camera, for one photo. It asks for the camera permission the first time (NSCameraUsageDescription); Pick a
/// photo needs none. From Examples/Look.
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

    func makeCoordinator() -> Coordinator { Coordinator(onPhoto: onPhoto, close: { dismiss() }) }

    @MainActor
    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onPhoto: @MainActor (UIImage) -> Void
        let close: @MainActor () -> Void

        init(onPhoto: @escaping @MainActor (UIImage) -> Void, close: @escaping @MainActor () -> Void) {
            self.onPhoto = onPhoto
            self.close = close
        }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage { onPhoto(image) }
            close()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { close() }
    }
}
#endif
