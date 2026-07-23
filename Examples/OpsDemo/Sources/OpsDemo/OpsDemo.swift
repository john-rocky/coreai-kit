// OpsDemo — the anchored ops end to end. Point it at a voice memo and one audio file
// becomes a transcript, a cleaned-up text, a summary, typed action items, a translation,
// and a spoken reply; point it at an image and it is captioned, object-detected, and
// OCR'd — all on device, no sessions or prompts in app code. Without arguments the text
// and search ops run on built-in sample text instead.
//
//   swift run OpsDemo                    # text + search ops on built-in samples
//   swift run OpsDemo voice-memo.wav     # full speech -> text -> speech pipeline
//   swift run OpsDemo photo.jpg          # caption + detect + document OCR

import CoreAIOps
import Foundation
import FoundationModels

@Generable
struct ActionItems {
    @Guide(description: "One entry per distinct task, as a short imperative sentence")
    var tasks: [String]
}

@Generable
struct Order {
    @Guide(description: "Name of the ordered product")
    var product: String
    @Guide(description: "Number of units ordered")
    var quantity: Int
    @Guide(description: "Requested delivery city")
    var city: String
}

let article = """
    Apple's FoundationModels framework gives every app a session API for on-device
    language models. CoreAIKit plugs downloadable open models into that same API, so the
    system model and your own models are interchangeable behind one code path. On top of
    it, CoreAIOps exposes task-level operations — summarize, extract, translate,
    proofread, transcribe — that hide sessions and prompts entirely: apps state the task,
    the kit resolves a catalog model behind it.
    """

let email = """
    Hi team, please ship 12 units of the AlphaWidget to our Osaka office by Friday.
    Thanks! — Dana
    """

let typos = "Their going to recieve the package tommorow, weather or not its raining."

let notes = [
    "Book flights to Sapporo for the snow festival and reserve a hotel near Odori Park.",
    "The quarterly report needs the updated revenue table before Thursday's review.",
    "Try the new ramen place by the station — open until 2am on weekends.",
    "Renew the Apple Developer membership before the certificate expires.",
]

@main
enum OpsDemo {
    static func main() async throws {
        // One hook covers every op's first-use download (~500 events/file, \r keeps one line).
        CoreAI.onDownload { progress in
            print(
                "\rdownloading \(progress.currentFile) \(Int(progress.fraction * 100))%",
                terminator: progress.fraction >= 1 ? "\n" : "")
        }
        guard let path = CommandLine.arguments.dropFirst().first else {
            try await textOps()
            return
        }
        let url = URL(fileURLWithPath: path)
        switch url.pathExtension.lowercased() {
        case "jpg", "jpeg", "png", "heic", "tiff", "webp":
            try await imagePipeline(url)
        default:
            try await voiceMemoPipeline(url)
        }
    }

    /// One voice memo in, six op results out — audio at both ends.
    static func voiceMemoPipeline(_ memo: URL) async throws {
        print("> CoreAI.transcribe(memo)")
        let text = try await CoreAI.transcribe(memo)
        print("[transcript] \(text)\n")

        print("> CoreAI.proofread(text)")
        let clean = try await CoreAI.proofread(text)
        print("[clean] \(clean)\n")

        print("> CoreAI.summarize(clean, style: .oneLine)")
        let summary = try await CoreAI.summarize(clean, style: .oneLine)
        print("[summary] \(summary)\n")

        print("> CoreAI.extract(clean, as: ActionItems.self)")
        let items = try await CoreAI.extract(clean, as: ActionItems.self)
        for task in items.tasks { print("[task] \(task)") }
        print()

        print("> CoreAI.translate(clean, to: .japanese)")
        let japanese = try await CoreAI.translate(clean, to: .japanese)
        print("[ja] \(japanese)\n")

        print("> CoreAI.speak(summary)")
        let audio = try await CoreAI.speak(summary)
        let out = URL(fileURLWithPath: "speech.wav")
        try writeWAV(audio, to: out)
        print("[audio] \(String(format: "%.1f", audio.seconds)) s -> \(out.path)")
    }

    /// One image in, three op results out.
    static func imagePipeline(_ url: URL) async throws {
        print("> CoreAI.caption(imageAt: url)")
        let caption = try await CoreAI.caption(imageAt: url)
        print("[caption] \(caption)\n")

        print("> CoreAI.detect(in: image)")
        let image = try ImageFile.load(url).cgImage
        let boxes = try await CoreAI.detect(in: image)
        for detection in boxes.prefix(8) {
            print("[object] \(detection.label)  \(String(format: "%.0f%%", detection.score * 100))")
        }
        if boxes.isEmpty { print("[object] none above threshold") }
        print()

        print("> CoreAI.read(documentAt: url)")
        let markdown = try await CoreAI.read(documentAt: url)
        print("[read]\n\(markdown)")
    }

    /// The text + search ops on built-in samples — no audio or image file needed.
    static func textOps() async throws {
        print("> CoreAI.summarize(article, style: .oneLine)")
        let summary = try await CoreAI.summarize(article, style: .oneLine)
        print("[summary] \(summary)\n")

        print("> CoreAI.extract(email, as: Order.self)")
        let order = try await CoreAI.extract(email, as: Order.self)
        print("[order] product=\(order.product) quantity=\(order.quantity) city=\(order.city)\n")

        print("> CoreAI.translate(email, to: .japanese)")
        let japanese = try await CoreAI.translate(email, to: .japanese)
        print("[ja] \(japanese)\n")

        print("> CoreAI.proofread(typos)")
        let clean = try await CoreAI.proofread(typos)
        print("[clean] \(clean)\n")

        print("> CoreAI.search(\"What should I plan for the trip?\", in: notes, topK: 2)")
        for hit in try await CoreAI.search("What should I plan for the trip?", in: notes, topK: 2) {
            print("[hit] \(String(format: "%.2f", hit.score))  \(hit.document)")
        }
    }

    /// Minimal 16-bit PCM mono WAV writer for the `speak` result.
    static func writeWAV(_ audio: SpokenAudio, to url: URL) throws {
        var data = Data()
        func tag(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let pcm = audio.samples.map { Int16((max(-1, min(1, $0)) * 32767).rounded()) }
        tag("RIFF"); u32(UInt32(36 + pcm.count * 2)); tag("WAVE")
        tag("fmt "); u32(16); u16(1); u16(1)
        u32(UInt32(audio.sampleRate)); u32(UInt32(audio.sampleRate * 2)); u16(2); u16(16)
        tag("data"); u32(UInt32(pcm.count * 2))
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
    }
}
