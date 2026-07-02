// VLChatModel — drives a local vision-language model through the FoundationModels stack:
// `LanguageModelSession(model: KitVisionModel)`. Pick a photo, ask about it, stream the
// answer. Everything (vision encode + decode) runs on-device.

import CoreAIKitUI
import CoreGraphics
import FoundationModels
import ImageIO
import Observation
import PhotosUI
import SwiftUI

/// A picked photo plus its EXIF orientation (camera photos arrive rotated).
struct PickedImage {
    let cgImage: CGImage
    let orientation: CGImagePropertyOrientation
}

@MainActor
@Observable
final class VLChatModel {
    enum Status: Equatable {
        case idle
        case downloading(Double)
        case loading
        case warming
        case ready
        case generating
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Pick a model and load it"
            case .downloading(let f): return "Downloading… \(Int(f * 100))%"
            case .loading: return "Loading…"
            case .warming: return "Warming up…"
            case .ready: return "Ready — pick a photo"
            case .generating: return "Thinking…"
            case .error(let message): return "Error: \(message)"
            }
        }
    }

    var status: Status = .idle
    /// Vision-language entries published for this platform (8B is Mac-only).
    var entries: [CatalogEntry] = []
    var selectedEntry: CatalogEntry?
    var bubbles: [ChatBubble] = []
    var picked: PickedImage?

    private var model: KitVisionModel?
    private var session: LanguageModelSession?
    private var generationTask: Task<Void, Never>?

    var isBusy: Bool {
        switch status {
        case .downloading, .loading, .warming, .generating: return true
        default: return false
        }
    }
    var downloadFraction: Double? {
        if case .downloading(let f) = status { return f }
        return nil
    }
    var canSend: Bool { status == .ready && picked != nil }

    // MARK: - Model loading

    /// Live catalog with the built-in snapshot as offline fallback.
    func loadCatalog() async {
        guard entries.isEmpty else { return }
        entries = await ModelCatalog.load().available(.vlm)
        if selectedEntry == nil { selectedEntry = entries.first }
    }

    func load() {
        guard !isBusy, let entry = selectedEntry else { return }
        status = .loading
        bubbles = []
        session = nil
        model = nil
        Task {
            do {
                // Same gesture as the model card: the catalog id resolves the platform
                // variant and the decoder + vision bundle pair.
                let model = try await KitVisionModel(catalog: entry.id) { progress in
                    Task { @MainActor in
                        self.status =
                            progress.fraction < 1 ? .downloading(progress.fraction) : .loading
                    }
                }
                self.model = model
                let session = LanguageModelSession(model: model)
                self.session = session
                status = .warming
                try await session.prewarm()
                status = .ready
            } catch {
                status = .error(error.localizedDescription)
            }
        }
    }

    // MARK: - Image

    func setImage(from item: PhotosPickerItem) {
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                let source = CGImageSourceCreateWithData(data as CFData, nil),
                let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else { return }
            let properties =
                CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
            let raw = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
            let orientation = CGImagePropertyOrientation(rawValue: raw) ?? .up
            picked = PickedImage(cgImage: cgImage, orientation: orientation)
            // A fresh photo starts a fresh conversation about it.
            newChat()
        }
    }

    // MARK: - Generation

    func send(_ text: String) {
        guard let session, let picked, status == .ready else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let question = trimmed.isEmpty ? "Describe this image in detail." : trimmed

        bubbles.append(ChatBubble(role: .user, content: question))
        bubbles.append(ChatBubble(role: .assistant, isStreaming: true))
        status = .generating

        let prompt = Prompt {
            question
            Attachment(picked.cgImage, orientation: picked.orientation)
        }
        generationTask = Task {
            do {
                let stream = session.streamResponse(
                    to: prompt, options: GenerationOptions(maximumResponseTokens: 320))
                for try await snapshot in stream {
                    if Task.isCancelled { break }
                    bubbles[bubbles.count - 1].content = snapshot.content
                }
            } catch {
                bubbles[bubbles.count - 1].content =
                    "⚠️ \(error.localizedDescription)"
            }
            bubbles[bubbles.count - 1].isStreaming = false
            if status == .generating { status = .ready }
        }
    }

    func stop() {
        generationTask?.cancel()
    }

    /// Fresh conversation on the loaded model (keeps the picked photo).
    func newChat() {
        guard let model else { return }
        generationTask?.cancel()
        bubbles = []
        session = LanguageModelSession(model: model)
        if status == .generating { status = .ready }
    }
}
