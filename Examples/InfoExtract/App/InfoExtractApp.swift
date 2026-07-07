// InfoExtract — on-device PII redaction demo over CoreAIKit's InformationExtractor (GLiNER2-PII).
// Paste text → the model detects entities (zero-shot, any label set) → the app redacts them in
// place and lists what it found, entirely on device (GPU / ANE), no network. Launch with the env
// var INFOEXTRACT_GATE=1 to instead run the headless demo-suite self-test (agent-verifiable gate).

import SwiftUI
import CoreAIKitEmbeddings

@main
struct InfoExtractApp: App {
    var body: some Scene { WindowGroup { RedactView() } }
}

private let PII_LABELS = ["person", "email", "phone number", "credit card number",
                          "social security number", "address", "date of birth", "organization"]

private let SAMPLE = """
Hi, this is Dr. Sarah Johnson — reach me at sarah.j@acme.com or +1-415-555-0142. \
Bill account holder James Lee, SSN 123-45-6789, card 4111 1111 1111 1111, and mail \
the receipt to 500 Market St, San Francisco. Treated 1985-07-14 at Mercy General Hospital.
"""

enum RedactStyle: String, CaseIterable, Identifiable {
    case label = "[LABEL]", block = "██ block"
    var id: String { rawValue }
    func replace(_ s: InformationExtractor.EntitySpan) -> String {
        switch self {
        case .label: return "[\(s.label.uppercased())]"
        case .block: return String(repeating: "█", count: max(3, s.text.count))
        }
    }
}

@MainActor
final class Model: ObservableObject {
    @Published var input = SAMPLE
    @Published var redacted = ""
    @Published var spans: [InformationExtractor.EntitySpan] = []
    @Published var style: RedactStyle = .label
    @Published var status = "loading model…"
    @Published var busy = false
    private var extractor: InformationExtractor?

    func load() async {
        do {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let t0 = Date()
            extractor = try await InformationExtractor(
                bundleAt: docs.appendingPathComponent("gliner2_bundle"))
            status = String(format: "ready (loaded in %.1fs)", Date().timeIntervalSince(t0))
            if ProcessInfo.processInfo.environment["INFOEXTRACT_GATE"] == "1" { await runGate() }
            else { await run() }
        } catch {
            status = "model not found — sideload gliner2_bundle into Documents. (\(error))"
        }
    }

    func run() async {
        guard let extractor, !busy else { return }
        busy = true; defer { busy = false }
        do {
            let t0 = Date()
            // ONE model run — extract spans, then redact from those spans locally. Running two
            // extractions concurrently on the shared graph corrupts it (every token → 0.50).
            let found = try await extractor.extractSpans(from: input, entities: PII_LABELS)
            redacted = InformationExtractor.redact(input, using: found,
                                                   replacement: { [style] in style.replace($0) })
            spans = found.sorted { $0.confidence > $1.confidence }
            status = String(format: "%d entities in %d ms", spans.count,
                            Int(Date().timeIntervalSince(t0) * 1000))
            print("[infoextract] run: \(spans.count) spans -> \(redacted)")
        } catch { status = "error: \(error)" }
    }

    // Headless demo-suite gate (INFOEXTRACT_GATE=1) — parity with Mac/Python ext.extract.
    func runGate() async {
        guard let extractor else { return }
        func log(_ s: String) { print(s); NSLog("[infoextract] %@", s) }
        let suite: [(String, [String: [String]])] = [
            ("Contact Dr. Sarah Johnson at sarah.j@acme.com or +1-415-555-0142.",
             ["person": ["Sarah Johnson"], "email": ["sarah.j@acme.com"], "phone number": ["+1-415-555-0142"]]),
            ("Wire to account holder James Lee, SSN 123-45-6789, card 4111 1111 1111 1111.",
             ["person": ["James Lee"], "credit card number": ["4111 1111 1111 1111"], "social security number": ["123-45-6789"]]),
            ("Send the invoice to 500 Market St, San Francisco and cc maria@corp.io.",
             ["email": ["maria@corp.io"], "address": ["500 Market St, San Francisco"]]),
            ("Patient DOB 1985-07-14, treated at Mercy General Hospital by nurse Alan Poe.",
             ["person": ["Alan Poe"], "date of birth": ["1985-07-14"], "organization": ["Mercy General Hospital"]]),
        ]
        var allOK = true
        for (t, expected) in suite {
            let got = (try? await extractor.extract(from: t, entities: PII_LABELS)) ?? [:]
            let ok = got == expected; allOK = allOK && ok
            log("\(ok ? "PASS" : "FAIL") \(t.prefix(46))")
        }
        log("GATE_RESULT: \(allOK ? "PASS" : "FAIL")")
        status = allOK ? "GATE PASS ✅" : "GATE FAIL ❌"
    }
}

struct RedactView: View {
    @StateObject private var m = Model()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Type or paste text — PII is detected and redacted on device.")
                        .font(.subheadline).foregroundStyle(.secondary)

                    TextEditor(text: $m.input)
                        .frame(minHeight: 130)
                        .padding(8)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                        .font(.callout)

                    HStack {
                        Picker("Style", selection: $m.style) {
                            ForEach(RedactStyle.allCases) { Text($0.rawValue).tag($0) }
                        }.pickerStyle(.segmented)
                        Button { Task { await m.run() } } label: {
                            Label("Redact", systemImage: "eye.slash")
                        }.buttonStyle(.borderedProminent).disabled(m.busy)
                    }
                    .onChange(of: m.style) { _, _ in Task { await m.run() } }

                    if m.busy { ProgressView().frame(maxWidth: .infinity) }

                    if !m.redacted.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("REDACTED").font(.caption).foregroundStyle(.secondary)
                            Text(m.redacted).font(.callout).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(.green.opacity(0.10), in: RoundedRectangle(cornerRadius: 10))
                        }
                    }

                    if !m.spans.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("DETECTED").font(.caption).foregroundStyle(.secondary)
                            ForEach(Array(m.spans.enumerated()), id: \.offset) { _, s in
                                HStack {
                                    Text(s.label).font(.caption2.weight(.semibold))
                                        .padding(.horizontal, 8).padding(.vertical, 3)
                                        .background(.tint.opacity(0.15), in: Capsule())
                                    Text(s.text).font(.callout)
                                    Spacer()
                                    Text(String(format: "%.2f", s.confidence))
                                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("PII Redactor")
            .safeAreaInset(edge: .bottom) {
                Text(m.status).font(.caption2).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity).padding(6).background(.bar)
            }
        }
        .task { await m.load() }
    }
}
