import CoreAIKit
import Foundation
import SwiftUI

/// Optional, repeatable first-use check on the same app and package as the UI.
/// Enable COREAI_ENTRY_SMOKE=1 in the scheme's environment. Every run gets its own
/// empty cache; the normal app cache is never removed or reused.
struct EntrySmokeView: View {
    @State private var output = "Starting first-use check…"

    static var isRequested: Bool {
        ProcessInfo.processInfo.environment["COREAI_ENTRY_SMOKE"] == "1"
    }

    var body: some View {
        ScrollView { Text(output).font(.system(.body, design: .monospaced)).padding() }
            .task { await run() }
    }

    @MainActor
    private func run() async {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let record = documents.appendingPathComponent("entry-check.txt")
        output = ""
        func log(_ line: String) {
            output += line + "\n"
            print(line)
            try? output.write(to: record, atomically: true, encoding: .utf8)
        }
        do {
            guard let model = ModelCatalog.builtin.entry(id: "qwen3-0.6b")?.modelID else {
                throw CoreAIKitError.modelNotAvailableOnPlatform(id: "qwen3-0.6b")
            }
            let cache = documents.appendingPathComponent("EntryChecks/\(UUID().uuidString)")
            let store = ModelStore(directory: cache)
            log("CoreAIKit 0.4.1 · qwen3-0.6b")
            log("OS \(ProcessInfo.processInfo.operatingSystemVersionString)")
            log("MODEL \(model.repo) @ \(model.revision) / \(model.resolvedPath)")
            log("EMPTY_CACHE \(store.localURL(for: model) == nil)")
            var configuration = ChatSession.Configuration()
            configuration.temperature = nil
            configuration.maxResponseTokens = 256
            let chat = try await ChatSession(model: model, store: store, configuration: configuration) { progress in
                print("DOWNLOAD \(progress.completedBytes)/\(progress.totalBytes)")
            }
            log("DOWNLOADED_AND_LOADED")
            let first = try await chat.respond(to: "Remember: my secret word is ORCHID. Confirm briefly. /no_think")
            log("TURN1 \(first)")
            let second = try await chat.respond(to: "What is my secret word? Reply with only that word. /no_think")
            log("TURN2 \(second)")
            guard !first.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  second.uppercased().contains("ORCHID") else {
                log("FAIL: expected a first reply and ORCHID on turn two")
                return
            }
            log("PASS: first download and two replies")
        } catch {
            log("FAIL: \(error)")
        }
    }
}
