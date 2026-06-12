import CoreAIKit
import CoreAIKitEmbeddings
import Foundation
import FoundationModels

struct IndexedChunk: Sendable {
    let file: String
    let text: String
    let vector: [Float]
}

struct SearchNotesTool: Tool {
    let name = "search_notes"
    let description = "Search the user's notes for passages relevant to a query."

    @Generable
    struct Arguments {
        @Guide(description: "What to look for, as a short search query")
        var query: String
    }

    let embedder: TextEmbedder
    let index: [IndexedChunk]

    func call(arguments: Arguments) async throws -> String {
        let queryVector = try await embedder.embed(query: arguments.query)
        let top = index
            .map { ($0, TextEmbedder.cosineSimilarity($0.vector, queryVector)) }
            .sorted { $0.1 > $1.1 }
            .prefix(3)
        let hits = top.map { String(format: "%@ %.2f", $0.0.file, $0.1) }
        print("  [tool] search_notes(\"\(arguments.query)\") → \(hits.joined(separator: ", "))")
        return top.map { "[\($0.0.file)]\n\($0.0.text)" }.joined(separator: "\n---\n")
    }
}

@main
enum DocChat {
    static func main() async throws {
        var args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else {
            print("usage: DocChat <notes-folder> [question]")
            print("env: KIT_EMBED_BUNDLE / KIT_CHAT_BUNDLE point at local bundles")
            return
        }
        let folder = URL(fileURLWithPath: args.removeFirst())
        let question = args.first ?? "What do my notes say about the bike trip?"
        let env = ProcessInfo.processInfo.environment

        // 1) Embedder + index.
        let embedder: TextEmbedder
        if let path = env["KIT_EMBED_BUNDLE"] {
            embedder = try await TextEmbedder(bundleAt: URL(fileURLWithPath: path))
        } else {
            print("Fetching EmbeddingGemma (cached afterwards)…")
            embedder = try await TextEmbedder()
        }

        var index: [IndexedChunk] = []
        let files = try FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil
        ).filter { ["md", "txt"].contains($0.pathExtension) }
        for file in files {
            let content = try String(contentsOf: file, encoding: .utf8)
            for chunk in chunks(of: content) {
                let vector = try await embedder.embed(document: chunk)
                index.append(
                    IndexedChunk(file: file.lastPathComponent, text: chunk, vector: vector))
            }
        }
        print("indexed \(index.count) chunks from \(files.count) files")
        guard !index.isEmpty else {
            print("no .md/.txt files found in \(folder.path)")
            return
        }

        // 2) Local LLM behind LanguageModelSession, with the retrieval tool.
        let model: KitLanguageModel
        if let path = env["KIT_CHAT_BUNDLE"] {
            model = try await KitLanguageModel(bundleAt: URL(fileURLWithPath: path))
        } else {
            print("Fetching qwen3 0.6B (cached afterwards)…")
            model = try await KitLanguageModel(model: .qwen3_0_6B)
        }
        let session = LanguageModelSession(
            model: model,
            tools: [SearchNotesTool(embedder: embedder, index: index)],
            instructions:
                "You answer questions about the user's notes. "
                + "Use search_notes to find relevant passages before answering; "
                + "ground your answer on what it returns."
        )

        print("\n> \(question)")
        let response = try await session.respond(to: question)
        print("\n[answer] \(response.content)")
    }

    /// Paragraph-merging chunker: split on blank lines, pack into ~500-char chunks.
    static func chunks(of text: String, target: Int = 500) -> [String] {
        let paragraphs = text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        var result: [String] = []
        var current = ""
        for paragraph in paragraphs {
            if current.isEmpty {
                current = paragraph
            } else if current.count + paragraph.count + 2 <= target {
                current += "\n\n" + paragraph
            } else {
                result.append(current)
                current = paragraph
            }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }
}
