// InboxModel — the screen's state: the inbox, the classifier, and the run as its answers arrive.
// The sort loop runs on InboxSorter's actor; answers reach the main actor in batches, at most 20 a
// second, so drawing never slows the loop. Every number on the screen is computed here from the
// run's own measurements.

import CoreAIKitEmbeddings
import Foundation
import Observation

@MainActor
@Observable
final class InboxModel {
    enum Phase: Equatable { case loading, ready, sorting, done, failed }

    let messages: [Message]
    let seed: UInt64
    private(set) var phase = Phase.loading
    /// LOADING: the download percent or "graphs"; FAILED: the error.
    private(set) var detail = ""
    /// By message index; nil until the message is sorted.
    private(set) var results: [SortedMessage?]
    private(set) var done = 0
    /// Median per-message time (ms) and messages per second, over the messages sorted so far.
    private(set) var medianMs: Double?
    private(set) var rate: Double?
    private(set) var counts: [String: [String: Int]] = [:]
    private(set) var sequenceLengthCounts: [Int: Int] = [:]
    /// When Sort was pressed; the clock runs from here.
    private(set) var startedAt: ContinuousClock.Instant?
    private(set) var run: SortRun?
    private(set) var loadSeconds: Double?
    private(set) var warmSeconds: [Int: Double] = [:]
    private(set) var sequenceLengths: [Int] = []
    private(set) var bundlePath = ""
    /// "bundle" (-bundle), "sideload" (Documents/gliner25-decide) or "catalog" (downloaded).
    private(set) var source = ""
    /// The graphs' precision, from classifier.json ("float16" -> "fp16").
    private(set) var precision = "fp16"
    private(set) var thermalBefore: ProcessInfo.ThermalState?
    private(set) var thermalAfter: ProcessInfo.ThermalState?

    private var sorter: InboxSorter?
    private var loadStarted = false
    private var milliseconds: [Double] = []
    private var runTask: Task<Void, Never>?

    init(count: Int, seed: UInt64) {
        self.seed = seed
        messages = Inbox.generate(count: count, seed: seed)
        results = Array(repeating: nil, count: messages.count)
    }

    var count: Int { messages.count }

    // MARK: - loading

    /// Finds the bundle, loads it, and runs one input through each graph (they load on first use).
    func load(bundle: String?) async {
        guard !loadStarted else { return }  // once, whichever window asks first
        loadStarted = true
        let t0 = ContinuousClock.now
        do {
            let url = try await resolveBundle(bundle)
            bundlePath = url.path
            precision = Self.precision(of: url)
            detail = "graphs"
            let classifier = try await TextClassifier(bundleAt: url)
            sequenceLengths = classifier.sequenceLengths
            let sorter = InboxSorter(classifier: classifier)
            warmSeconds = try await sorter.warm()
            self.sorter = sorter
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
            source = "bundle"
            return URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath, isDirectory: true)
        }
        #if os(iOS)
        let sideload = URL.documentsDirectory.appending(path: "gliner25-decide", directoryHint: .isDirectory)
        if FileManager.default.fileExists(atPath: sideload.appending(path: "classifier.json").path) {
            source = "sideload"
            return sideload
        }
        #endif
        source = "catalog"
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

    /// Sorts the whole inbox from the first message; DONE starts it over with the labels cleared.
    func sort() {
        guard let sorter, phase == .ready || phase == .done else { return }
        reset()
        let start = ContinuousClock.now
        startedAt = start
        thermalBefore = ProcessInfo.processInfo.thermalState
        phase = .sorting
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

    // MARK: - what the screen and the log read

    /// Seconds on the clock: from Sort to now while sorting, the run's total once done.
    func elapsed(at now: ContinuousClock.Instant) -> Double? {
        switch phase {
        case .sorting: return startedAt.map { $0.duration(to: now).inSeconds }
        case .done: return run?.totalSeconds
        default: return nil
        }
    }

    /// The screen's state as one line, for the autoplay log.
    var statusLine: String {
        switch phase {
        case .loading: return "LOADING \(detail)"
        case .ready: return String(format: "READY %d messages · load %.2f s", count, loadSeconds ?? 0)
        case .sorting:
            return "SORTING \(done)/\(count) · " + Self.ms(medianMs) + " per message · " + Self.rate(rate)
        case .done:
            let ms = run?.milliseconds ?? []
            return String(format: "DONE %d/%d · %.2f s · median %@ · p90 %@ · %@", done, count, run?.totalSeconds ?? 0,
                          Self.ms(Stats.percentile(ms, 0.5)), Self.ms(Stats.percentile(ms, 0.9)), Self.rate(rate))
        case .failed: return "FAILED \(detail)"
        }
    }

    static func ms(_ v: Double?) -> String {
        guard let v else { return "– ms" }
        return v < 10 ? String(format: "%.1f ms", v) : String(format: "%.0f ms", v)
    }

    static func rate(_ v: Double?) -> String {
        guard let v else { return "– msg/s" }
        return String(format: "%.1f msg/s", v)
    }

    /// The finished run as the result file's JSON: the CLI's summary plus where it ran.
    func resultJSON() -> JSONValue? {
        guard let run else { return nil }
        var out = run.summary()
        out["device"] = .string(Device.model)
        out["machine"] = .string(Device.machine)
        out["os"] = .string(Device.os)
        out["os_build"] = .string(ProcessInfo.processInfo.operatingSystemVersionString)
        out["bundle"] = .string(bundlePath)
        out["source"] = .string(source)
        out["precision"] = .string(precision)
        out["compute_units"] = .string("gpu")
        out["seed"] = .int(Int(seed))
        out["tasks"] = .object(Dictionary(uniqueKeysWithValues: InboxTasks.all.map {
            ($0.name, JSONValue.array($0.labels.map { .string($0) }))
        }))
        out["load_s"] = .rounded(loadSeconds ?? 0, 3)
        out["warm_s"] = .object(Dictionary(uniqueKeysWithValues: warmSeconds.map { ("\($0.key)", .rounded($0.value, 3)) }))
        out["duplicates"] = .int(count - Set(messages.map(\.text)).count)
        out["agree_with_written"] = .object(run.agreement(with: messages).mapValues { .rounded($0, 3) })
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
