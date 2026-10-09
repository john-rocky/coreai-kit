// InboxModel — the screen's state: where the inbox comes from (pasted text, a file, the sample
// inbox), the categories it is sorted into, the classifier, and the run as its answers arrive. The
// model loads once at launch whatever the source; the inbox and the categories can be changed
// before Sort and again after DONE (a change clears the answers). The sort loop runs on
// InboxSorter's actor; answers reach the main actor in batches, at most 20 a second, so drawing
// never slows the loop. Every number on the screen is computed here from the run's own measurements.

import CoreAIKitEmbeddings
import Foundation
import Observation
import UniformTypeIdentifiers

@MainActor
@Observable
final class InboxModel {
    enum Phase: Equatable { case loading, ready, sorting, done, failed }

    /// Where the inbox comes from.
    enum Source: String, CaseIterable, Sendable {
        case paste, file, sample

        /// The picker's words.
        var title: String {
            switch self {
            case .paste: return "Paste"
            case .file: return "Import file"
            case .sample: return "Sample inbox"
            }
        }

        /// What the footer and the DONE line call the inbox.
        var inboxName: String {
            switch self {
            case .paste: return "pasted messages"
            case .file: return "your file"
            case .sample: return "sample inbox"
            }
        }
    }

    /// The two ways the answers are shown.
    enum Showing: String, CaseIterable, Sendable {
        case inbox, category

        var title: String { self == .inbox ? "Inbox" : "By category" }
    }

    /// The files Import file offers: plain text, CSV and Markdown.
    static let fileTypes: [UTType] = [.plainText, .commaSeparatedText, UTType("net.daringfireball.markdown"), .text]
        .compactMap { $0 }

    /// The sample inbox: the generator's messages from `seed`, the inbox `textclassify-cli --inbox`
    /// sorts.
    let sampleCount: Int
    let seed: UInt64
    private(set) var source: Source
    /// The inbox on screen, in order.
    private(set) var messages: [Message] = []
    /// Why there is nothing to sort, or how many pieces were skipped.
    private(set) var note: String?
    /// Pasted or imported pieces with no letter or digit, for the inbox on screen.
    private(set) var skipped = 0
    var showing: Showing
    private(set) var phase = Phase.loading
    /// LOADING: the download percent or "graphs"; FAILED: the error.
    private(set) var detail = ""

    /// The categories field, and what it reads as.
    var categoriesText: String
    private(set) var categories: InboxCategories.Parsed
    /// The categories of the answers on screen (the run's).
    private(set) var sortedCategories: [String] = []

    // The Paste sheet: its text is kept between openings.
    var showingPaste = false
    var pasteText = ""
    /// Why the sheet's last Sort was refused, until the text changes.
    private(set) var pasteRefusal: String?
    /// The file picker of Import file.
    var importing = false

    /// By message index; nil until the message is sorted.
    private(set) var results: [SortedMessage?] = []
    private(set) var done = 0
    /// Median per-message time (ms) and messages per second, over the messages sorted so far.
    private(set) var medianMs: Double?
    private(set) var rate: Double?
    private(set) var counts: [String: [String: Int]] = [:]
    private(set) var sequenceLengthCounts: [Int: Int] = [:]
    /// When Sort was pressed; the clock runs from here.
    private(set) var startedAt: ContinuousClock.Instant?
    private(set) var run: SortRun?
    /// From launch to READY: finding (or downloading) the bundle, loading it, and one run through
    /// each graph.
    private(set) var loadSeconds: Double?
    private(set) var warmSeconds: [Int: Double] = [:]
    private(set) var sequenceLengths: [Int] = []
    private(set) var bundlePath = ""
    /// "bundle" (-bundle), "sideload" (Documents/gliner25-decide) or "catalog" (downloaded).
    private(set) var modelSource = ""
    /// The graphs' precision, from classifier.json ("float16" -> "fp16").
    private(set) var precision = "fp16"
    private(set) var thermalBefore: ProcessInfo.ThermalState?
    private(set) var thermalAfter: ProcessInfo.ThermalState?

    private var classifier: TextClassifier?
    private var loadStarted = false
    private var begun = false
    private var milliseconds: [Double] = []
    private var runTask: Task<Void, Never>?
    /// Held from Sort to DONE (see `load`).
    private var sortActivity: (any NSObjectProtocol)?

    init(source: Source, sampleCount: Int, seed: UInt64, categories: String?, showing: Showing) {
        self.source = source
        self.sampleCount = sampleCount
        self.seed = seed
        self.showing = showing
        let text = categories ?? InboxCategories.defaultText
        categoriesText = text
        self.categories = InboxCategories.parse(text)
    }

    var count: Int { messages.count }

    /// The categories the bars and By category show: the run's once Sort was pressed, else the
    /// field's (none while the field cannot be read).
    var shownCategories: [String] {
        run != nil || phase == .sorting ? sortedCategories : categories.labels
    }

    /// Whether Sort can be pressed now, and if not, why (nil when it can or when it is not offered).
    var sortRefusal: String? {
        if let problem = categories.problem { return problem.message }
        if messages.isEmpty { return note ?? InboxInput.Problem.empty.message }
        return nil
    }

    var canSort: Bool {
        classifier != nil && (phase == .ready || phase == .done) && sortRefusal == nil
    }

    // MARK: - the inbox

    /// The inbox the app opens on: the sample inbox, the Paste sheet, or for a file the empty card.
    func begin() {
        guard !begun else { return }  // once, whichever window asks
        begun = true
        switch source {
        case .sample: setInbox(Inbox.generate(count: sampleCount, seed: seed))
        case .paste:
            setInbox([], note: "Nothing pasted yet")
            showingPaste = true
        case .file: setInbox([], note: "No file picked yet")
        }
    }

    /// Paste opens the sheet and Import file the file picker (the inbox on screen stays until they
    /// give a new one); Sample inbox puts the sample on screen. Not while sorting.
    func select(_ source: Source) {
        guard phase != .sorting else { return }
        switch source {
        case .paste: showingPaste = true
        case .file: importing = true
        case .sample:
            self.source = .sample
            setInbox(Inbox.generate(count: sampleCount, seed: seed))
        }
    }

    private func setInbox(_ inbox: [Message], skipped: Int = 0, note: String? = nil) {
        messages = inbox
        self.skipped = skipped
        self.note = note ?? (skipped > 0 ? InboxInput.skippedNote(skipped) : nil)
        clearAnswers()
    }

    /// The answers on screen no longer match the inbox or the categories: back to READY.
    private func clearAnswers() {
        reset()
        run = nil
        sortedCategories = []
        if phase == .done { phase = .ready }
    }

    // MARK: - the Paste sheet

    /// The sheet's text as it would be read.
    var pasteReading: InboxInput.Reading { InboxInput.read(text: pasteText) }

    /// The sheet's line under the text: "100 messages", "98 messages · 2 skipped (…)", or why
    /// nothing can be sorted.
    var pasteStatus: String {
        if let pasteRefusal { return pasteRefusal }
        let reading = pasteReading
        if let problem = reading.problem { return problem.message }
        var line = Self.messagesLabel(reading.messages.count)
        if reading.skipped > 0 { line += " · " + InboxInput.skippedNote(reading.skipped) }
        return line
    }

    /// The sheet's Sort: a readable paste goes on screen and, once the model is ready and the
    /// categories read, is sorted at once; a refusal stays in the sheet. Returns whether the sheet
    /// closed.
    @discardableResult
    func submitPaste() -> Bool {
        guard phase != .sorting else { return false }
        let reading = pasteReading
        guard reading.problem == nil else {
            pasteRefusal = reading.note
            return false
        }
        pasteRefusal = nil
        source = .paste
        setInbox(InboxInput.messages(reading.messages), skipped: reading.skipped)
        showingPaste = false
        if canSort { sort() }
        return true
    }

    /// The text changed: an earlier refusal no longer applies.
    func pasteEdited() { pasteRefusal = nil }

    /// Import file: the file's messages go on screen, or the reason they cannot.
    func importFile(_ url: URL) {
        guard phase != .sorting else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let reading = InboxInput.read(contentsOf: url)
        source = .file
        if let problem = reading.problem {
            setInbox([], skipped: reading.skipped, note: problem.message)
        } else {
            setInbox(InboxInput.messages(reading.messages), skipped: reading.skipped)
        }
    }

    // MARK: - the categories

    /// The field changed: read it again, and clear answers given under other categories.
    func categoriesEdited() {
        let parsed = InboxCategories.parse(categoriesText)
        guard parsed != categories else { return }
        categories = parsed
        if phase == .done { clearAnswers() }
    }

    // MARK: - loading

    /// Finds the bundle, loads it, and runs one input through each graph (they load on first use).
    func load(bundle: String?) async {
        guard !loadStarted else { return }  // once, whichever window asks first
        loadStarted = true
        // The download, the load and a sort are work a person asked for. Without this, macOS App Naps a window
        // that is hidden or behind others: a first download fell to 0.13 MB/s on a Mac whose curl got 5 MB/s.
        let activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Loading the model")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        let t0 = ContinuousClock.now
        do {
            let url = try await resolveBundle(bundle)
            bundlePath = url.path
            precision = Self.precision(of: url)
            detail = "graphs"
            let classifier = try await TextClassifier(bundleAt: url)
            sequenceLengths = classifier.sequenceLengths
            warmSeconds = try await InboxSorter(classifier: classifier).warm()
            self.classifier = classifier
            loadSeconds = t0.duration(to: .now).inSeconds
            detail = ""
            phase = .ready
        } catch {
            detail = error.localizedDescription
            phase = .failed
        }
    }

    /// `-bundle <dir>`; else, on an iPhone, a copy sideloaded into Documents/gliner25-decide; else
    /// the catalog's gliner2.5-decide at its pinned revision, downloaded once and cached.
    private func resolveBundle(_ explicit: String?) async throws -> URL {
        if let explicit {
            modelSource = "bundle"
            return URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath, isDirectory: true)
        }
        #if os(iOS)
        let sideload = URL.documentsDirectory.appending(path: "gliner25-decide", directoryHint: .isDirectory)
        if FileManager.default.fileExists(atPath: sideload.appending(path: "classifier.json").path) {
            modelSource = "sideload"
            return sideload
        }
        #endif
        modelSource = "catalog"
        detail = "0%"
        let entry = try await ModelCatalog.entry(forID: "gliner2.5-decide", expecting: .textClassification)
        guard let id = entry.modelID else { throw CoreAIKitError.modelNotAvailableOnPlatform(id: entry.id) }
        return try await ModelStore.default.download(id) { progress in
            Task { @MainActor in
                if self.phase == .loading, self.detail != "graphs" {
                    self.detail = "\(Int(progress.fraction * 100))%"
                }
            }
        }
    }

    private static func precision(of bundle: URL) -> String {
        struct Config: Decodable { let dtype: String? }
        let dtype = (try? JSONDecoder().decode(
            Config.self, from: Data(contentsOf: bundle.appending(path: "classifier.json"))))?.dtype
        switch dtype {
        case "float16": return "fp16"
        case "float32": return "fp32"
        case let other?: return other
        case nil: return "fp16"
        }
    }

    // MARK: - sorting

    /// Sorts the whole inbox from the first message into the field's categories; DONE starts it
    /// over with the answers cleared.
    func sort() {
        guard canSort, let classifier else { return }
        reset()
        sortedCategories = categories.labels
        let sorter = InboxSorter(classifier: classifier, tasks: InboxCategories.tasks(sortedCategories))
        let start = ContinuousClock.now
        startedAt = start
        thermalBefore = ProcessInfo.processInfo.thermalState
        phase = .sorting
        sortActivity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Sorting the inbox")
        let messages = self.messages
        runTask = Task {
            let (batches, sink) = AsyncStream.makeStream(of: [SortedMessage].self)
            let loop = Task.detached(priority: .userInitiated) { () async throws -> SortRun in
                defer { sink.finish() }
                return try await sorter.sort(messages, start: start) { sink.yield($0) }
            }
            for await batch in batches { apply(batch) }
            do {
                finish(try await loop.value)
            } catch {
                detail = error.localizedDescription
                phase = .failed
            }
            if let sortActivity { ProcessInfo.processInfo.endActivity(sortActivity) }
            sortActivity = nil
        }
    }

    private func reset() {
        results = Array(repeating: nil, count: messages.count)
        done = 0
        milliseconds = []
        medianMs = nil
        rate = nil
        counts = [:]
        sequenceLengthCounts = [:]
        startedAt = nil
        run = nil
        thermalAfter = nil
    }

    private func apply(_ batch: [SortedMessage]) {
        guard let last = batch.last else { return }
        var results = self.results
        var counts = self.counts
        for r in batch {
            results[r.index] = r
            for (task, label) in r.labels { counts[task, default: [:]][label, default: 0] += 1 }
            sequenceLengthCounts[r.sequenceLength, default: 0] += 1
            milliseconds.append(r.seconds * 1000)
        }
        self.results = results
        self.counts = counts
        done += batch.count
        medianMs = Stats.percentile(milliseconds.sorted(), 0.5)
        if last.at > 0 { rate = Double(done) / last.at }
    }

    private func finish(_ run: SortRun) {
        self.run = run
        medianMs = Stats.percentile(run.milliseconds, 0.5)
        rate = run.messagesPerSecond
        thermalAfter = ProcessInfo.processInfo.thermalState
        phase = .done
    }

    // MARK: - By category

    /// Urgency from the heaviest: critical, high, normal, low.
    static let urgencyOrder = ["critical", "high", "normal", "low"]

    /// Every category with its sorted messages, the heaviest urgency first, then in inbox order.
    var byCategory: [(category: String, items: [(message: Message, result: SortedMessage)])] {
        let rank = Dictionary(uniqueKeysWithValues: Self.urgencyOrder.enumerated().map { ($0.element, $0.offset) })
        var groups: [String: [(message: Message, result: SortedMessage)]] = [:]
        for m in messages {
            guard m.id < results.count, let r = results[m.id], let c = r.labels["intent"] else { continue }
            groups[c, default: []].append((m, r))
        }
        return shownCategories.map { c in
            let items = (groups[c] ?? []).sorted {
                let a = rank[$0.result.labels["urgency"] ?? ""] ?? 4, b = rank[$1.result.labels["urgency"] ?? ""] ?? 4
                return a != b ? a < b : $0.message.id < $1.message.id
            }
            return (c, items)
        }
    }

    // MARK: - what the screen and the log read

    /// Seconds on the clock: from Sort to now while sorting, the run's total once done.
    func elapsed(at now: ContinuousClock.Instant) -> Double? {
        switch phase {
        case .sorting: return startedAt.map { $0.duration(to: now).inSeconds }
        case .done: return run?.totalSeconds
        default: return nil
        }
    }

    /// The screen's state as one line, for the autoplay log. It never holds a message's text.
    var statusLine: String {
        var line: String
        switch phase {
        case .loading: return "LOADING \(detail)"
        case .ready:
            line = String(format: "READY %d messages · source %@ · categories %d · skipped %d · load %.2f s · warm %@",
                          count, source.rawValue, categories.labels.count, skipped, loadSeconds ?? 0, warmLine)
        case .sorting:
            return "SORTING \(done)/\(count) · " + Self.ms(medianMs) + " per message · " + Self.rate(rate)
        case .done:
            let ms = run?.milliseconds ?? []
            line = String(format: "DONE %d/%d · %.2f s · median %@ · p90 %@ · %@ · source %@ · %@", done, count,
                          run?.totalSeconds ?? 0, Self.ms(Stats.percentile(ms, 0.5)), Self.ms(Stats.percentile(ms, 0.9)),
                          Self.rate(rate), source.rawValue, showing == .inbox ? "Inbox" : "By category")
        case .failed: return "FAILED \(detail)"
        }
        if let note { line += " · note: \(note)" }
        if let problem = categories.problem { line += " · categories: \(problem.message)" }
        if showingPaste { line += " · paste sheet: " + pasteStatus }
        return line
    }

    /// "S=256 0.12 s, S=512 0.20 s".
    var warmLine: String {
        warmSeconds.keys.sorted().map { String(format: "S=%d %.2f s", $0, warmSeconds[$0]!) }.joined(separator: ", ")
    }

    static func messagesLabel(_ n: Int) -> String { "\(grouped(n)) message\(n == 1 ? "" : "s")" }

    static func ms(_ v: Double?) -> String {
        guard let v else { return "– ms" }
        return v < 10 ? String(format: "%.1f ms", v) : String(format: "%.0f ms", v)
    }

    static func rate(_ v: Double?) -> String {
        guard let v else { return "– msg/s" }
        return String(format: "%.1f msg/s", v)
    }

    /// The finished run as the result file's JSON: the CLI's summary plus where the inbox came from,
    /// the categories, every message's answers (never its text) and where it ran.
    func resultJSON() -> JSONValue? {
        guard let run else { return nil }
        var out = run.summary()
        out["source"] = .string(source.rawValue)
        out["categories"] = .array(sortedCategories.map { .string($0) })
        out["messages_skipped"] = .int(skipped)
        out["device"] = .string(Device.model)
        out["machine"] = .string(Device.machine)
        out["os"] = .string(Device.os)
        out["os_build"] = .string(ProcessInfo.processInfo.operatingSystemVersionString)
        out["bundle"] = .string(bundlePath)
        out["model_source"] = .string(modelSource)
        out["precision"] = .string(precision)
        out["compute_units"] = .string("gpu")
        if source == .sample { out["seed"] = .int(Int(seed)) }
        out["tasks"] = .object(Dictionary(uniqueKeysWithValues: run.tasks.map {
            ($0.name, JSONValue.array($0.labels.map { .string($0) }))
        }))
        out["load_s"] = .rounded(loadSeconds ?? 0, 3)
        out["warm_s"] = .object(Dictionary(uniqueKeysWithValues: warmSeconds.map { ("\($0.key)", .rounded($0.value, 3)) }))
        out["duplicates"] = .int(count - Set(messages.map(\.text)).count)
        // The generator's own labels exist only for the sample inbox, and only the default
        // categories are the labels it wrote.
        if source == .sample, sortedCategories == InboxCategories.defaults {
            out["agree_with_written"] = .object(run.agreement(with: messages).mapValues { .rounded($0, 3) })
        }
        // Every message's answers and their probabilities, in inbox order: what the parity of two
        // runs is read from.
        out["answers"] = .array(run.results.map { r in
            var a: [String: JSONValue] = ["i": .int(r.index)]
            for t in run.tasks {
                a[t.name] = .string(r.labels[t.name] ?? "")
                a["p_" + t.name] = .rounded(Double(r.confidence[t.name] ?? 0), 4)
            }
            if r.truncated { a["truncated"] = .bool(true) }
            return .object(a)
        })
        // Every message's [seconds after Sort, milliseconds, sequence length], in inbox order: the
        // raw series behind the medians, so a reader can recompute them and see the rate over time.
        out["series"] = .array(run.results.map {
            .array([.rounded($0.at, 3), .rounded($0.seconds * 1000, 2), .int($0.sequenceLength)])
        })
        out["thermal_before"] = .string(Device.name(thermalBefore))
        out["thermal_after"] = .string(Device.name(thermalAfter))
        out["timestamp"] = .string(ISO8601DateFormatter().string(from: Date()))
        return .object(out)
    }
}

/// Where the run happened, for the footer and the result file.
enum Device {
    /// `utsname.machine`: the model identifier on an iPhone, the CPU architecture on a Mac.
    static let machine: String = {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: &u.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }()

    /// The model identifier (iPhone18,1 / Mac16,10).
    static let model: String = {
        #if os(macOS)
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var buffer = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &buffer, &size, nil, 0)
        return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
        #else
        return machine
        #endif
    }()

    /// "iOS 27.0" / "macOS 27.0".
    static let os: String = {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        #if os(iOS)
        let name = "iOS"
        #else
        let name = "macOS"
        #endif
        let version = v.patchVersion > 0 ? "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)" : "\(v.majorVersion).\(v.minorVersion)"
        return "\(name) \(version)"
    }()

    static func name(_ state: ProcessInfo.ThermalState?) -> String {
        switch state {
        case .nominal?: return "nominal"
        case .fair?: return "fair"
        case .serious?: return "serious"
        case .critical?: return "critical"
        default: return "unknown"
        }
    }
}
