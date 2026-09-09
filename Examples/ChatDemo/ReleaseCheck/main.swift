import CoreAIKit
import Foundation
import FoundationModels

func log(_ text: String) {
    FileHandle.standardOutput.write(Data((text + "\n").utf8))
}
func progress(_ p: DownloadProgress) {
    log("DOWNLOAD \(p.completedBytes)/\(p.totalBytes) \(p.currentFile)")
}
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "EntryCheck", code: 1,
        userInfo: [NSLocalizedDescriptionKey: message]) }
}
let args = CommandLine.arguments
let mode = args.count > 1 ? args[1] : "chat"
let root = URL(fileURLWithPath: args.count > 2 ? args[2] : ".models")
let store = ModelStore(directory: root)
let entry = ModelCatalog.builtin.entry(id: "qwen3-0.6b")!
let id = entry.modelID!
if !["speak", "download-tts", "hybrid"].contains(mode) {
    log("IDENTITY \(id.repo) revision=\(id.revision) path=\(id.resolvedPath) cached=\(store.localURL(for: id) != nil)")
}
do {
    switch mode {
    case "download":
        let url = try await store.download(id, progress: progress)
        log("DOWNLOADED \(url.path)")
        let cached = try await store.download(id) { _ in log("UNEXPECTED_CACHE_DOWNLOAD") }
        try require(cached == url, "cache path changed")
        log("CACHE_HIT \(cached.path)")
    case "download-tts":
        let tts = try await ModelCatalog.entry(forID: "voxcpm-0.5b", expecting: .tts)
        for path in [tts.variant!.path, "tokenizer", "voxcpm_host_glue"] {
            let model = tts.modelID(path: path)
            log("TTS_IDENTITY \(model.repo) revision=\(model.revision) path=\(path)")
            _ = try await store.download(model, progress: progress)
        }
    case "chat":
        let chat = try await ChatSession(model: id, store: store, downloadProgress: progress)
        log("LOADED")
        var answer = ""
        for try await event in await chat.streamResponse(to: "Hello!") {
            if case .response(let delta) = event {
                if answer.isEmpty { log("FIRST_OUTPUT \(delta)") }
                answer += delta
            }
        }
        log("ANSWER \(answer)")
        try require(!answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "empty chat answer")
    case "fm":
        let model = try await KitLanguageModel(model: id, store: store, downloadProgress: progress)
        let session = LanguageModelSession(model: model)
        log("LOADED")
        let first = try await session.respond(to: "Remember: my secret word is ORCHID. Reply with one short sentence confirming it.")
        log("TURN1 \(first.content)")
        let second = try await session.respond(to: "What is my secret word? Reply with only that word.")
        log("TURN2 \(second.content)")
        try require(!first.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "empty FM turn one")
        try require(second.content.uppercased().contains("ORCHID"), "FM turn two did not recall ORCHID")
        log("TRANSCRIPT_ENTRIES \(session.transcript.count)")
    case "japanese":
        var configuration = ChatSession.Configuration()
        configuration.temperature = nil
        configuration.maxResponseTokens = 1024
        let chat = try await ChatSession(model: id, store: store, configuration: configuration)
        var answer = ""
        var chunks = 0
        var completed = ""
        log("LOADED")
        for try await event in await chat.streamResponse(to: "日本語だけで、春の京都を散歩する旅行者への案内を400文字程度で書いてください。桜、お寺、食べ物、交通について紹介してください。/no_think") {
            switch event {
            case .response(let delta):
                chunks += 1
                answer += delta
                log("DELTA \(chunks) \(delta)")
            case .complete(let message): completed = message.content
            default: break
            }
        }
        log("ANSWER \(answer)")
        log("STREAM chunks=\(chunks) characters=\(answer.count) completeMatches=\(answer == completed)")
        let japanese = answer.unicodeScalars.filter {
            (0x3040...0x30FF).contains(Int($0.value)) || (0x4E00...0x9FFF).contains(Int($0.value))
        }.count
        try require(japanese > 100, "not enough Japanese output")
        try require(chunks > 1 && answer == completed, "stream did not match complete response")
        try require(!answer.contains("\u{FFFD}"), "replacement character in Japanese stream")
    case "japanese-fm":
        let model = try await KitLanguageModel(model: id, store: store)
        let session = LanguageModelSession(model: model)
        log("LOADED")
        var answer = ""
        var snapshots = 0
        for try await snapshot in session.streamResponse(
            to: "日本語だけで、春の京都を散歩する旅行者への案内を400文字程度で書いてください。桜、お寺、食べ物、交通について紹介してください。/no_think",
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 1024)) {
            snapshots += 1
            answer = snapshot.content
            log("SNAPSHOT \(snapshots) characters=\(answer.count) \(answer)")
        }
        log("ANSWER \(answer)")
        log("STREAM snapshots=\(snapshots) characters=\(answer.count)")
        let japanese = answer.unicodeScalars.filter {
            (0x3040...0x30FF).contains(Int($0.value)) || (0x4E00...0x9FFF).contains(Int($0.value))
        }.count
        try require(japanese > 100, "not enough Japanese FM output")
        try require(snapshots > 1, "FM stream produced no intermediate snapshots")
        try require(!answer.contains("\u{FFFD}"), "replacement character in Japanese FM stream")
    case "hybrid":
        let hybrid = ModelCatalog.builtin.entry(id: "qwen3.5-0.8b")!
        let model = hybrid.modelID!
        log("HYBRID_IDENTITY \(model.repo) revision=\(model.revision) path=\(model.resolvedPath) cached=\(store.localURL(for: model) != nil)")
        var configuration = ChatSession.Configuration()
        configuration.temperature = nil
        configuration.maxResponseTokens = 512
        configuration.engineVariant = .pipelined
        let chat = try await ChatSession(model: model, store: store, configuration: configuration, downloadProgress: progress)
        log("LOADED")
        let first = try await chat.respond(to: "Remember: my secret word is ORCHID. Reply with one short sentence confirming it. /no_think")
        log("TURN1 \(first)")
        let second = try await chat.respond(to: "What is my secret word? Reply with only that word. /no_think")
        log("TURN2 \(second)")
        try require(!first.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "empty hybrid turn one")
        try require(second.uppercased().contains("ORCHID"), "hybrid turn two did not recall ORCHID")
        log("HISTORY_ENTRIES \(await chat.history.count)")
    case "speak":
        let tts = try await ModelCatalog.entry(forID: "voxcpm-0.5b", expecting: .tts)
        try require(tts.revision == ModelCatalog.builtin.entry(id: "voxcpm-0.5b")?.revision,
                    "live TTS catalog changed; see the release validation record before reproducing")
        log("TTS_IDENTITY \(tts.repo) revision=\(tts.revision ?? "unpinned")")
        let speaker = try await KitSpeaker(catalog: "voxcpm-0.5b", store: store, downloadProgress: progress)
        log("LOADED")
        let audio = try await speaker.synthesize("Hello from Core AI.")
        try require(audio.sampleRate == 16000 && !audio.samples.isEmpty, "empty or wrong-rate audio")
        try require(audio.samples.allSatisfy(\.isFinite), "non-finite PCM")
        let peak = audio.samples.map { abs($0) }.max() ?? 0
        let rms = sqrt(audio.samples.reduce(0.0) { $0 + Double($1 * $1) } / Double(audio.samples.count))
        try require(peak > 0.001 && rms > 0.0001, "silent audio")
        let wav = root.deletingLastPathComponent().appendingPathComponent("speech.wav")
        try WAVFile.write(samples: audio.samples, sampleRate: audio.sampleRate, to: wav)
        log("AUDIO samples=\(audio.samples.count) rate=\(audio.sampleRate) seconds=\(audio.seconds) peak=\(peak) rms=\(rms) wav=\(wav.path)")
    default: throw NSError(domain: "EntryCheck", code: 2)
    }
    log("PASS \(mode)")
} catch {
    log("FAIL \(mode): \(error)")
    exit(1)
}
