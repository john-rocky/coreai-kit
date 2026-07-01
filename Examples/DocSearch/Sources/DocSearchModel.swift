import CoreAIKitEmbeddings
import CoreGraphics
import Foundation
import Observation
import UIKit

// On-device visual document retrieval over a corpus of page images (bundled samples + any you
// import). Each page is encoded as a 3×3 grid of tiles with ColModernVBERT's document encoder;
// a typed query is encoded with the query encoder; pages are ranked by MaxSim over all tiles.
// The best-matching tile localizes *where* on the page the query matched (reliable because tiles
// are spatially-disjoint inputs — unlike within-tile patch tokens).
@MainActor
@Observable
final class DocSearchModel {
    // Tiles per SHORT image edge; the long edge gets proportionally more so each tile is ~square
    // (a uniform N×N grid on a portrait page yields tall tiles + more squish distortion). Higher
    // → tighter highlight, but more encoder calls per page at index time.
    static let tilesShortSide = 4

    // Aspect-aware grid: ~square tiles regardless of page orientation.
    static func grid(for image: UIImage) -> (rows: Int, cols: Int) {
        let w = max(image.size.width, 1), h = max(image.size.height, 1)
        let n = tilesShortSide
        if w <= h {
            return (rows: max(1, Int((Double(n) * h / w).rounded())), cols: n)
        } else {
            return (rows: n, cols: max(1, Int((Double(n) * w / h).rounded())))
        }
    }

    struct Page: Identifiable, Hashable {
        let id = UUID()
        let title: String
        let image: UIImage
        var embedding: VisualDocumentRetriever.TiledPageEmbedding?

        static func == (lhs: Page, rhs: Page) -> Bool { lhs.id == rhs.id }
        func hash(into hasher: inout Hasher) { hasher.combine(id) }
    }

    struct Hit: Identifiable {
        let id = UUID()
        let page: Page
        let score: Float
    }

    enum Phase: Equatable {
        case starting
        case downloading(Double)
        case indexing(done: Int, total: Int)
        case ready
        case error(String)

        var label: String {
            switch self {
            case .starting: return "Loading ColModernVBERT…"
            case .downloading(let f): return "Fetching model… \(Int(f * 100))%"
            case .indexing(let d, let t): return "Indexing pages \(d)/\(t)…"
            case .ready: return "Ready"
            case .error(let m): return "Error: \(m)"
            }
        }
    }

    var phase: Phase = .starting
    var pages: [Page] = []
    var results: [Hit] = []
    var searchMilliseconds: Double?
    var currentQuery = ""

    private var retriever: VisualDocumentRetriever?
    private var lastQueryEmbedding: [[Float]]?

    var isReady: Bool { if case .ready = phase { return true } else { return false } }

    func start() async {
        guard retriever == nil else { return }
        pages = Self.loadBundledPages()
        do {
            let retriever = try await VisualDocumentRetriever { progress in
                Task { @MainActor in self.phase = .downloading(progress.fraction) }
            }
            self.retriever = retriever
            for index in pages.indices {
                phase = .indexing(done: index, total: pages.count)
                if let cg = pages[index].image.cgImage {
                    let g = Self.grid(for: pages[index].image)
                    pages[index].embedding = try await retriever.encodeTiled(
                        page: cg, rows: g.rows, cols: g.cols)
                }
            }
            phase = .ready
        } catch {
            phase = .error(error.localizedDescription)
        }
    }

    func addImportedImage(_ image: UIImage, title: String) async {
        guard let retriever, let cg = image.cgImage else { return }
        do {
            var page = Page(title: title, image: image)
            let g = Self.grid(for: image)
            page.embedding = try await retriever.encodeTiled(
                page: cg, rows: g.rows, cols: g.cols)
            pages.append(page)
            if !currentQuery.isEmpty { await search(currentQuery) }
        } catch {
            phase = .error(error.localizedDescription)
        }
    }

    func search(_ query: String) async {
        guard let retriever else { return }
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        currentQuery = trimmed
        guard !trimmed.isEmpty else {
            results = []; lastQueryEmbedding = nil; searchMilliseconds = nil
            return
        }
        do {
            let start = SuspendingClock.now
            let q = try await retriever.encode(query: trimmed)
            lastQueryEmbedding = q
            let scored: [Hit] = pages.compactMap { page in
                guard let emb = page.embedding else { return nil }
                return Hit(page: page, score: retriever.score(query: q, tiledPage: emb))
            }
            .sorted { $0.score > $1.score }
            let elapsed = (SuspendingClock.now - start).components
            searchMilliseconds =
                Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
            results = scored
        } catch {
            phase = .error(error.localizedDescription)
        }
    }

    /// The page region (normalized rect, origin top-left) best matching the current query, or nil.
    func bestTileRect(for page: Page) -> CGRect? {
        guard let retriever, let q = lastQueryEmbedding, let emb = page.embedding else { return nil }
        return retriever.bestTile(query: q, tiledPage: emb)
    }

    private static func loadBundledPages() -> [Page] {
        let urls = (Bundle.main.urls(forResourcesWithExtension: "png", subdirectory: nil) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("doc_") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return urls.compactMap { url in
            guard let img = UIImage(contentsOfFile: url.path) else { return nil }
            let key = url.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: "doc_", with: "")
            return Page(title: key.capitalized, image: img)
        }
    }
}
