// ModelCatalog.swift — a remote, updatable list of known-good hosted models, so new
// models reach apps without a package update. Fetched from the repo's catalog.json with
// a built-in snapshot as offline fallback.

import Foundation

public struct CatalogEntry: Sendable, Identifiable, Codable, Hashable {
    public enum Kind: String, Sendable, Codable {
        case chat
        case imageText
        case depth
        case textEmbedding
    }

    public struct Variant: Sendable, Codable, Hashable {
        /// Subtree inside the repo holding the bundle.
        public let path: String
        /// Approximate download size, for UI.
        public let sizeMB: Int?

        public init(path: String, sizeMB: Int? = nil) {
            self.path = path
            self.sizeMB = sizeMB
        }
    }

    public let id: String
    public let name: String
    public let repo: String
    public let kind: Kind
    /// Keyed by platform: "macos" / "ios". A missing key = not published there.
    public let variants: [String: Variant]
    public let thinking: Bool?

    public init(
        id: String, name: String, repo: String, kind: Kind,
        variants: [String: Variant], thinking: Bool? = nil
    ) {
        self.id = id
        self.name = name
        self.repo = repo
        self.kind = kind
        self.variants = variants
        self.thinking = thinking
    }

    static var platformKey: String {
        #if os(iOS)
        return "ios"
        #else
        return "macos"
        #endif
    }

    /// The variant for this platform, or nil if the model is not published here.
    public var variant: Variant? { variants[Self.platformKey] }

    /// `ModelID` for this platform, or nil if the model is not published here.
    public var modelID: ModelID? {
        variant.map { ModelID(repo, path: $0.path) }
    }
}

public struct ModelCatalog: Sendable, Codable {
    public let version: Int
    public let models: [CatalogEntry]

    public init(version: Int, models: [CatalogEntry]) {
        self.version = version
        self.models = models
    }

    /// Entries available on this platform, optionally filtered by kind.
    public func available(_ kind: CatalogEntry.Kind? = nil) -> [CatalogEntry] {
        models.filter { entry in
            entry.modelID != nil && (kind == nil || entry.kind == kind)
        }
    }

    public static let defaultURL = URL(
        string: "https://raw.githubusercontent.com/john-rocky/coreai-kit/main/catalog.json")!

    public static func fetch(from url: URL = defaultURL) async throws -> ModelCatalog {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw CoreAIKitError.httpError(statusCode: -1, file: url.lastPathComponent)
        }
        return try JSONDecoder().decode(ModelCatalog.self, from: data)
    }

    /// Fetches the live catalog, falling back to the built-in snapshot — never throws.
    public static func load(from url: URL = defaultURL) async -> ModelCatalog {
        (try? await fetch(from: url)) ?? .builtin
    }

    /// Snapshot of the hosted models at packaging time (mirrors catalog.json).
    public static let builtin = ModelCatalog(
        version: 1,
        models: [
            CatalogEntry(
                id: "qwen3-0.6b", name: "Qwen3 0.6B",
                repo: "mlboydaisuke/qwen3-0.6b-CoreAI-official", kind: .chat,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 352),
                    "ios": .init(path: "ios", sizeMB: 454),
                ],
                thinking: true),
            CatalogEntry(
                id: "qwen3-4b", name: "Qwen3 4B",
                repo: "mlboydaisuke/qwen3-4b-CoreAI-official", kind: .chat,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 2280),
                    "ios": .init(path: "ios", sizeMB: 2498),
                ],
                thinking: true),
            CatalogEntry(
                id: "mistral-7b-v0.3", name: "Mistral 7B v0.3",
                repo: "mlboydaisuke/mistral-7b-v0.3-CoreAI-official", kind: .chat,
                variants: ["macos": .init(path: "macos", sizeMB: 4078)]),
            CatalogEntry(
                id: "gemma-3-4b-it", name: "Gemma 3 4B",
                repo: "mlboydaisuke/gemma-3-4b-it-CoreAI-official", kind: .chat,
                variants: ["macos": .init(path: "macos", sizeMB: 2223)]),
            CatalogEntry(
                id: "clip-vit-b32", name: "CLIP ViT-B/32",
                repo: "mlboydaisuke/clip-vit-base-patch32-CoreAI-official", kind: .imageText,
                variants: [
                    "macos": .init(path: "model", sizeMB: 291),
                    "ios": .init(path: "model", sizeMB: 291),
                ]),
            CatalogEntry(
                id: "depth-anything-3-small", name: "Depth Anything 3 Small",
                repo: "mlboydaisuke/depth-anything-3-small-CoreAI-official", kind: .depth,
                variants: [
                    "macos": .init(path: "model", sizeMB: 101),
                    "ios": .init(path: "model", sizeMB: 101),
                ]),
            CatalogEntry(
                id: "embeddinggemma-300m", name: "EmbeddingGemma 300m",
                repo: "mlboydaisuke/embeddinggemma-300m-CoreAI", kind: .textEmbedding,
                variants: [
                    "macos": .init(path: "model", sizeMB: 1230),
                    "ios": .init(path: "model", sizeMB: 1230),
                ]),
        ])
}
