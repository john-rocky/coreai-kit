// CoreAI+Capability.swift — the question the kit could not be asked.
//
// `CoreAI.prepare(.transcribe)` is an instruction: fetch this now. Before an app can decide
// whether to *offer* a feature it needs the question — can this happen, and what will it cost
// the user — and there was no way to ask. So every adopter either shipped a button that fails
// on a full disk, or wrote the check themselves out of `catalog.json`.
//
// ```swift
// switch await CoreAI.capability(.transcribeMeeting) {
// case .ready:                      showButton()
// case .needsDownload(let bytes):   showPrompt(bytes)          // "Meeting notes needs 238 MB"
// case .needsSystemAssets:          showFirstRunNotice()
// case .insufficientStorage(let needs, let free): showFreeSpace(needs - free)
// case .unsupportedDevice(let why): hideFeature(why)
// }
// ```
//
// Why none of this surfaced earlier: it does not fire on the developer's machine. The device
// is supported, the disk has room, and the model is cached from the last run. It fires on
// users' devices, after shipping.
//
// **Answers offline, and does not touch the network.** A picker showing every op must not
// become 20 round trips, and a capability check in `.task { }` must work in airplane mode. The
// size comes from the catalog; `ModelStore.remoteSize(of:)` is there for a caller that wants
// the exact figure and can afford to ask.

import CoreAIKit
import Foundation

/// What it would take to run an op right now.
public enum Capability: Sendable, Equatable {
    /// Everything needed is on the device. The call will just run.
    case ready
    /// Ready after a download of this size, which counts against the app.
    case needsDownload(bytes: Int64)
    /// Ready after the OS fetches assets of its own — a locale pack for the system
    /// transcriber. Distinct from `needsDownload` because the bytes are not the app's: they
    /// are shared with every other app on the device, do not appear in the app's storage
    /// figure, and are usually already there. Worth a first-run notice, not a size prompt.
    case needsSystemAssets(what: String)
    /// The download does not fit.
    case insufficientStorage(needsBytes: Int64, freeBytes: Int64)
    /// This device cannot run it, whatever the disk says.
    case unsupportedDevice(reason: String)

    /// Bytes the app would have to download, zero when there is nothing to fetch.
    public var downloadBytes: Int64 {
        if case .needsDownload(let bytes) = self { return bytes }
        return 0
    }

    /// Whether the feature can be offered at all — true for everything except an unsupported
    /// device, since the rest are prompts rather than dead ends.
    public var isAvailable: Bool {
        if case .unsupportedDevice = self { return false }
        return true
    }
}

extension CoreAI {
    /// Headroom over the download size before storage is called sufficient. A bundle needs
    /// room to land *and* room to be unpacked and compiled, and a device that ends a download
    /// with nothing free is a device about to misbehave in other ways.
    static let storageHeadroom: Int64 = 500_000_000

    /// Can this op run, and what would it cost.
    ///
    /// Never throws: "I could not tell" is an honest `unsupportedDevice(reason:)` rather than
    /// an error a caller has to `catch` in the middle of building a view.
    public static func capability(
        _ op: Op, options: OpOptions = OpOptions(), store: ModelStore = .default
    ) async -> Capability {
        // Ops that now run on an Apple backend by default. Asking for a specific model
        // overrides it, which is what `options.model` means everywhere else.
        if options.model == nil, op == .transcribe || op == .transcribeMeeting {
            let speech = await systemSpeechCapability()
            // `transcribeMeeting` still needs the diarizer, which is a real download and the
            // more actionable answer when both are outstanding: the locale pack is the OS's
            // problem and is usually already there.
            if op == .transcribe { return speech }
            let diarizer = await catalogCapability(
                id: MeetingDiarizerID, store: store)
            if case .needsDownload = diarizer { return diarizer }
            if case .insufficientStorage = diarizer { return diarizer }
            if case .unsupportedDevice = diarizer { return diarizer }
            return speech
        }

        guard let id = op.defaultModelID.map({ options.model ?? $0 }) ?? options.model else {
            // The GLiNER2 ops load a pinned preset rather than a catalog entry. It is small
            // and always available; saying `ready` is truer than inventing a size.
            return .ready
        }
        return await catalogCapability(id: id, store: store)
    }

    /// The diarizer behind `transcribeMeeting` — the one speech capability Apple ships nothing
    /// for, and therefore the only part of that op that is a real download.
    static let MeetingDiarizerID = "sortformer-diar-v2"

    private static func catalogCapability(id: String, store: ModelStore) async -> Capability {
        guard let entry = ModelCatalog.builtin.entry(id: id) else {
            return .unsupportedDevice(
                reason: "no model with catalog id '\(id)' in this build of the kit")
        }
        guard let model = entry.modelID, let variant = entry.variant else {
            return .unsupportedDevice(
                reason: "'\(id)' is not published for this platform")
        }
        if store.isCached(model) { return .ready }

        // The catalog's figure, not the Hub's: this must answer offline and instantly. It is
        // the total a first run pulls, including sibling subtrees the loader resolves, which
        // is why it is used here rather than a per-path measurement.
        let needs = Int64(variant.sizeMB ?? 0) * 1_000_000
        if let free = store.availableBytes, needs > 0, free < needs + storageHeadroom {
            return .insufficientStorage(needsBytes: needs, freeBytes: free)
        }
        return .needsDownload(bytes: needs)
    }

    /// Capability for every op at once, for a gallery or a settings screen. Still no network.
    public static func capabilities(store: ModelStore = .default) async -> [Op: Capability] {
        var result: [Op: Capability] = [:]
        for op in Op.allCases {
            result[op] = await capability(op, store: store)
        }
        return result
    }

    /// Apple's transcriber, against the device's own locale.
    ///
    /// `transcribe` moved to `SpeechAnalyzer`, so the honest answer to "how big is
    /// transcription" changed from 3.2 GB to nothing. A capability API that could not say so
    /// would be reporting the wrong number for the op adopters reach for first.
    private static func systemSpeechCapability(
        locale: Locale = .current
    ) async -> Capability {
        guard await SystemTranscriber.isAvailable(for: locale) else {
            return .unsupportedDevice(
                reason: "the system transcriber does not support \(locale.identifier) on this "
                    + "device — pass options: .model(\"whisper-large-v3-turbo\") to use a "
                    + "catalog model instead")
        }
        return await SystemTranscriber.isInstalled(for: locale)
            ? .ready
            : .needsSystemAssets(what: "the \(locale.identifier) speech assets")
    }
}
