import CoreAIKit
import CoreGraphics
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
import WebKit

@MainActor
@Observable
final class ReadDocModel {
    var markdown = ""
    var status = "Pick a document image."
    var busy = false
    /// The source image, shown next to the OCR output so the transcription can be verified.
    var sourceImage: CGImage?
    /// OCR entries in the catalog; the picker reloads on a new choice.
    var entries: [CatalogEntry] = []
    var selectedEntry: CatalogEntry?

    private var reader: KitDocReader?
    private var mineruReader: KitMineruReader?
    private var glmReader: KitGlmOcrReader?
    private var loadedID: String?

    /// Live catalog (built-in snapshot offline) UNION the built-in snapshot, so locally-known
    /// models (e.g. MinerU2.5-Pro, sideloaded) still appear even when the remote catalog omits them.
    func loadCatalog() async {
        guard entries.isEmpty else { return }
        let live = await ModelCatalog.load().available(.ocr)
        let builtin = ModelCatalog.builtin.available(.ocr)
        let liveIDs = Set(live.map(\.id))
        entries = live + builtin.filter { !liveIDs.contains($0.id) }
        if selectedEntry == nil { selectedEntry = entries.first }
    }

    func read(_ url: URL) async {
        guard let entry = selectedEntry, !busy else { return }
        busy = true
        defer { busy = false }
        do {
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }

            // Show the source image immediately (before OCR) so the user sees what's being read.
            sourceImage = Self.loadCGImage(url)
            markdown = ""

            let t0 = Date()
            if entry.id.hasPrefix("mineru") {
                // MinerU rides the VL rope-shift rider (KitMineruReader) from AOT bundles under
                // Documents/mineru_pf/: {vision,decoder} = 768 recognition; layout/{vision,decoder}
                // = 1036² square Layout Detection. When the layout bundle is present the reader runs
                // the 2-stage pipeline (structured Markdown, tables → <table> HTML); otherwise a
                // single whole-page pass. (On-device JIT can't compile these graphs — AOT required.)
                if mineruReader == nil || loadedID != entry.id {
                    status = "Loading \(entry.name)…"
                    let docs = FileManager.default.urls(
                        for: .documentDirectory, in: .userDomainMask)[0]
                    let base = docs.appendingPathComponent("mineru_pf")
                    let layoutVision = base.appendingPathComponent("layout/vision")
                    let layoutDecoder = base.appendingPathComponent("layout/decoder")
                    let hasLayout = FileManager.default.fileExists(atPath: layoutVision.path)
                    mineruReader = try await KitMineruReader(
                        visionDir: base.appendingPathComponent("vision"),
                        decoderDir: base.appendingPathComponent("decoder"),
                        layoutVisionDir: hasLayout ? layoutVision : nil,
                        layoutDecoderDir: hasLayout ? layoutDecoder : nil)
                    reader = nil
                    loadedID = entry.id
                }
                glmReader = nil
                if mineruReader!.supportsStructured {
                    status = "Reading… (layout → regions)"
                    markdown = try await mineruReader!.readStructured(imageAt: url)
                } else {
                    status = "Reading…"
                    markdown = try await mineruReader!.read(imageAt: url)
                }
            } else if entry.id.hasPrefix("glm") {
                // GLM-OCR rides the same VL rope-shift rider (KitGlmOcrReader) from bundles
                // sideloaded into Documents/glm_ocr_pf/{vision,decoder}.
                if glmReader == nil || loadedID != entry.id {
                    status = "Loading \(entry.name)…"
                    let docs = FileManager.default.urls(
                        for: .documentDirectory, in: .userDomainMask)[0]
                    let base = docs.appendingPathComponent("glm_ocr_pf")
                    glmReader = try await KitGlmOcrReader(
                        visionDir: base.appendingPathComponent("vision"),
                        decoderDir: base.appendingPathComponent("decoder"))
                    reader = nil
                    mineruReader = nil
                    loadedID = entry.id
                }
                status = "Reading…"
                markdown = try await glmReader!.read(imageAt: url)
            } else {
                if reader == nil || loadedID != entry.id {
                    status = "Loading \(entry.name)…"
                    // Same gesture as the model card: the catalog id resolves the bundles.
                    reader = try await KitDocReader(catalog: entry.id) { progress in
                        Task { @MainActor in
                            self.status = "Downloading… \(Int(progress.fraction * 100))%"
                        }
                    }
                    mineruReader = nil
                    glmReader = nil
                    loadedID = entry.id
                }
                status = "Reading…"
                markdown = try await reader!.read(imageAt: url)
            }
            status = String(
                format: "Done — %d chars, %.1fs.", markdown.count, Date().timeIntervalSince(t0))
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    /// Decode a picked image file to a CGImage for display (cross-platform, via ImageIO).
    private static func loadCGImage(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}

struct ReadDocView: View {
    @State private var model = ReadDocModel()
    @State private var showImporter = false
    @State private var zoomImage = false
    /// Render the output as formatted Markdown/HTML (tables draw as tables) vs. the raw source.
    @State private var rendered = true

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                Picker("Model", selection: $model.selectedEntry) {
                    ForEach(model.entries) { entry in
                        Text(entry.name).tag(Optional(entry))
                    }
                }
                .fixedSize()
                Button("Open Image…") { showImporter = true }
                    .disabled(model.busy || model.selectedEntry == nil)
                Spacer()
                Picker("", selection: $rendered) {
                    Text("Rendered").tag(true)
                    Text("Raw").tag(false)
                }
                .pickerStyle(.segmented)
                .fixedSize()
                .disabled(model.markdown.isEmpty)
            }
            .padding(.horizontal, 12)

            // Source image + OCR output, side by side (wide) or stacked (phone) so the
            // transcription can be checked against the page. Tap the image to zoom.
            comparison

            HStack {
                Text(model.status)
                if model.busy { ProgressView().controlSize(.small) }
                Spacer()
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 12)
        }
        .padding(.vertical, 8)
        .task { await model.loadCatalog() }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result {
                Task { await model.read(url) }
            }
        }
        #if os(macOS)
        .sheet(isPresented: $zoomImage) { zoomCover }
        #else
        .fullScreenCover(isPresented: $zoomImage) { zoomCover }
        #endif
    }

    /// Full-screen (iOS) / sheet (macOS) zoom of the source image.
    @ViewBuilder private var zoomCover: some View {
        if let cg = model.sourceImage {
            ZoomableImageView(cgImage: cg) { zoomImage = false }
                .frame(minWidth: 400, minHeight: 400)
        }
    }

    /// Adaptive: image beside the text on a wide window, stacked on a narrow one. Keyed on the
    /// available *width* (not content) so Raw and Rendered lay out the same — a long unbroken
    /// `<table>` line in Raw must not force the panes to stack.
    private var comparison: some View {
        GeometryReader { geo in
            if geo.size.width > 620 {
                HStack(alignment: .top, spacing: 8) {
                    imagePane.frame(maxWidth: .infinity, maxHeight: .infinity)
                    textPane.frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                VStack(spacing: 8) {
                    imagePane.frame(maxHeight: 300)
                    textPane.frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .padding(.horizontal, 8)
    }

    private var imagePane: some View {
        Group {
            if let cg = model.sourceImage {
                // Pinch / drag / double-tap zoom INLINE (clipped to the pane — it's fine if the
                // page overflows the box while zoomed in). The ⤢ button opens the full screen.
                ZoomableImage(cgImage: cg)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.white)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(.gray.opacity(0.3)))
                    .overlay(alignment: .bottomTrailing) {
                        Button { zoomImage = true } label: {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                                .font(.caption).padding(6)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                        .buttonStyle(.plain)
                        .padding(6)
                    }
            } else {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color.gray.opacity(0.08))
                    .overlay(Text("Pick a document image →").foregroundStyle(.secondary))
            }
        }
    }

    private var textPane: some View {
        Group {
            if model.markdown.isEmpty {
                ScrollView {
                    Text("The document's markdown appears here.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                }
            } else if rendered {
                // Formatted view: paragraphs + real tables (the OCR emits <table> HTML).
                MarkdownWebView(html: Self.renderedHTML(model.markdown))
            } else {
                ScrollView {
                    Text(model.markdown)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                }
            }
        }
        .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    /// Convert a Markdown pipe table (GLM-OCR's table format) into an HTML `<table>`; the `:---`
    /// alignment row is dropped.
    static func pipeTableHTML(_ md: String, escape: (String) -> String) -> String {
        var html = "<table>"
        for line in md.split(separator: "\n") {
            let raw = line.trimmingCharacters(in: .whitespaces)
            guard raw.hasPrefix("|") else { continue }
            var cells = raw.split(separator: "|", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            if cells.first == "" { cells.removeFirst() }
            if cells.last == "" { cells.removeLast() }
            if !cells.isEmpty,
                cells.allSatisfy({ !$0.isEmpty && $0.allSatisfy { ":-".contains($0) } }) {
                continue  // separator row
            }
            html += "<tr>" + cells.map { "<td>\(escape($0))</td>" }.joined() + "</tr>"
        }
        return html + "</table>"
    }

    /// Wrap the reader's output (plain-text blocks, `#` headings, `<table>` HTML or Markdown pipe
    /// tables, `\n\n`-separated) in a small styled HTML document for the Rendered view.
    static func renderedHTML(_ markdown: String) -> String {
        func escape(_ s: String) -> String {
            s.replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;")
                .replacingOccurrences(of: "\n", with: "<br>")
        }
        var body = ""
        for block in markdown.components(separatedBy: "\n\n") {
            let t = block.trimmingCharacters(in: .whitespacesAndNewlines)
            if t.isEmpty { continue }
            if t.lowercased().hasPrefix("<table") {
                body += t                                   // MinerU: HTML table
            } else if t.hasPrefix("|"), t.contains("\n") {
                body += pipeTableHTML(t, escape: escape)    // GLM-OCR: Markdown pipe table
            } else if t.hasPrefix("## ") {
                body += "<h2>\(escape(String(t.dropFirst(3))))</h2>"
            } else if t.hasPrefix("# ") {
                body += "<h1>\(escape(String(t.dropFirst(2))))</h1>"
            } else {
                body += "<p>\(escape(t))</p>"
            }
        }
        let css = """
        :root { color-scheme: light dark; }
        body { font: 15px -apple-system, system-ui, sans-serif; margin: 14px;
               line-height: 1.5; -webkit-text-size-adjust: 100%; }
        h1 { font-size: 1.5em; margin: 4px 0 12px; }
        h2 { font-size: 1.2em; margin: 18px 0 8px; }
        p { margin: 0 0 12px; }
        table { border-collapse: collapse; margin: 8px 0 16px; width: 100%; font-size: 14px; }
        td, th { border: 1px solid rgba(128,128,128,0.5); padding: 5px 9px; text-align: left;
                 vertical-align: top; }
        tr:first-child td { font-weight: 600; background: rgba(128,128,128,0.14); }
        """
        return """
        <!doctype html><html><head>\
        <meta name="viewport" content="width=device-width, initial-scale=1">\
        <style>\(css)</style></head><body>\(body)</body></html>
        """
    }
}

/// A minimal cross-platform `WKWebView` that renders a self-contained HTML string, reloading only
/// when the HTML changes (so scroll position isn't lost on unrelated view updates).
struct MarkdownWebView {
    let html: String

    final class Coordinator { var lastHTML = "" }
    func makeCoordinator() -> Coordinator { Coordinator() }

    private func makeWebView() -> WKWebView {
        let web = WKWebView()
        #if os(macOS)
        web.setValue(false, forKey: "drawsBackground")
        #else
        web.isOpaque = false
        web.backgroundColor = .clear
        web.scrollView.backgroundColor = .clear
        #endif
        return web
    }

    private func reload(_ web: WKWebView, _ context: Coordinator) {
        guard context.lastHTML != html else { return }
        context.lastHTML = html
        web.loadHTMLString(html, baseURL: nil)
    }
}

#if os(macOS)
extension MarkdownWebView: NSViewRepresentable {
    func makeNSView(context: Context) -> WKWebView { makeWebView() }
    func updateNSView(_ web: WKWebView, context: Context) { reload(web, context.coordinator) }
}
#else
extension MarkdownWebView: UIViewRepresentable {
    func makeUIView(context: Context) -> WKWebView { makeWebView() }
    func updateUIView(_ web: WKWebView, context: Context) { reload(web, context.coordinator) }
}
#endif

/// A pinch-to-zoom / drag-to-pan / double-tap image. Zooms IN PLACE — the caller clips it to
/// whatever frame it occupies (inline pane or full screen), so the page may overflow when zoomed.
struct ZoomableImage: View {
    let cgImage: CGImage
    var maxScale: CGFloat = 40
    @State private var scale: CGFloat = 1
    @State private var lastScale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero

    var body: some View {
        Image(decorative: cgImage, scale: 1)
            .resizable()
            .scaledToFit()
            .scaleEffect(scale)
            .offset(offset)
            .gesture(
                SimultaneousGesture(
                    MagnificationGesture()
                        .onChanged { scale = max(1, min(maxScale, lastScale * $0)) }
                        .onEnded { _ in lastScale = scale },
                    DragGesture()
                        .onChanged {
                            offset = CGSize(
                                width: lastOffset.width + $0.translation.width,
                                height: lastOffset.height + $0.translation.height)
                        }
                        .onEnded { _ in lastOffset = offset }
                )
            )
            // Double-tap toggles fit ⇄ 4× (re-centres); a second double-tap resets.
            .onTapGesture(count: 2) {
                withAnimation {
                    if scale > 1 {
                        scale = 1; lastScale = 1; offset = .zero; lastOffset = .zero
                    } else {
                        scale = 4; lastScale = 4; offset = .zero; lastOffset = .zero
                    }
                }
            }
    }
}

/// Full-screen wrapper around `ZoomableImage`, so fine text can be checked against the OCR.
struct ZoomableImageView: View {
    let cgImage: CGImage
    let onClose: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            ZoomableImage(cgImage: cgImage)
            VStack {
                HStack {
                    Spacer()
                    Button(action: onClose) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.largeTitle)
                            .foregroundStyle(.white.opacity(0.9))
                            .padding()
                    }
                }
                Spacer()
                Text("Pinch to zoom (to 40×) · drag to pan · double-tap for 4× / reset")
                    .font(.caption).foregroundStyle(.white.opacity(0.6)).padding(.bottom, 24)
            }
        }
    }
}
