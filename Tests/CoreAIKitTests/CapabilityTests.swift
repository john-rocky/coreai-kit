// CapabilityTests.swift — the query an app branches on before offering a feature.
//
// The failure this guards against is not a crash. It is a capability check that quietly says
// "no" on a device where the feature works, because nobody files a report about a button they
// never saw. One of those was already found by running this on a real machine: a device set to
// English-in-Japan reports `en_JP`, which is in nobody's supported locale list, and strict
// matching declared transcription unsupported while ten English locales sat installed.

import Foundation
import Testing

@testable import CoreAIKit
@testable import CoreAIKitCore
@testable import CoreAIOps

struct CapabilityTests {
    /// Every op must resolve to something the kit can answer for. An op absent from the
    /// mapping reports "nothing to download", which is the most expensive possible wrong
    /// answer — it is the number that goes into a shipping decision.
    @Test func everyOpAnswers() async {
        for op in CoreAI.Op.allCases {
            let capability = await CoreAI.capability(op)
            if case .unsupportedDevice(let reason) = capability {
                // Legitimate on macOS for iOS-only entries, but it must say why.
                #expect(!reason.isEmpty, "\(op) is unsupported with no reason")
            }
        }
    }

    /// The mapping `capability`, `prepare` and `coreai-doctor` all walk. An op that is in the
    /// enum but has no model behind it silently costs nothing, which is how `watch` reported
    /// zero for an app that needed 103 MB.
    @Test func everyModelBackedOpNamesACatalogEntry() {
        let modelless: Set<CoreAI.Op> = [.redact, .extractEntities]
        for op in CoreAI.Op.allCases where !modelless.contains(op) {
            guard let id = op.defaultModelID else {
                Issue.record("\(op) has no defaultModelID and is not on the modelless list")
                continue
            }
            #expect(ModelCatalog.builtin.entry(id: id) != nil,
                    "\(op) names '\(id)', which is not in the catalog")
        }
    }

    @Test func liveOpsAreCoveredNow() {
        // The regression: these were added to the enum after the doctor reported an app using
        // `watch()` as having nothing to download.
        #expect(CoreAI.Op.watch.defaultModelID == CoreAI.defaultDetectionModel)
        #expect(CoreAI.Op.scan.defaultModelID == CoreAI.defaultDetectionModel)
        #expect(CoreAI.Op.watchDepth.defaultModelID == CoreAI.defaultDepthModel)
    }

    @Test func downloadBytesIsZeroUnlessThereIsADownload() {
        #expect(Capability.ready.downloadBytes == 0)
        #expect(Capability.needsSystemAssets(what: "x").downloadBytes == 0)
        #expect(Capability.needsDownload(bytes: 42).downloadBytes == 42)
    }

    @Test func onlyAnUnsupportedDeviceHidesTheFeature() {
        #expect(Capability.ready.isAvailable)
        #expect(Capability.needsDownload(bytes: 1).isAvailable)
        #expect(Capability.needsSystemAssets(what: "x").isAvailable)
        // Out of room is a prompt, not a dead end — the user can free space.
        #expect(Capability.insufficientStorage(needsBytes: 2, freeBytes: 1).isAvailable)
        #expect(!Capability.unsupportedDevice(reason: "old").isAvailable)
    }

    /// Sizes go in front of a person deciding whether to tap. They are rounded the way a
    /// person says them, and the unit changes at a gigabyte.
    @Test func errorSizesReadLikeSentences() {
        #expect(CoreAIKitError.gigabytes(3_300_000_000) == "3.3 GB")
        #expect(CoreAIKitError.gigabytes(238_000_000) == "238 MB")
        #expect(CoreAIKitError.gigabytes(1_000_000_000) == "1.0 GB")
    }

    /// The three failures a shipped app actually hits must have their own vocabulary. An
    /// engineer should never see `notAHuggingFaceRepo` from `CoreAI.summarize`.
    @Test func theShippedFailuresHaveTheirOwnErrors() {
        let storage = CoreAIKitError.insufficientStorage(
            needsBytes: 3_300_000_000, freeBytes: 100_000_000)
        let text = storage.errorDescription ?? ""
        #expect(text.contains("3.3 GB"))
        #expect(text.contains("100 MB"))
        #expect(text.contains("capability"), "the error should name the way to avoid it")

        #expect(CoreAIKitError.unsupportedDevice(reason: "no ANE").errorDescription?
            .contains("no ANE") == true)
        #expect(CoreAIKitError.systemAssetsUnavailable(what: "the ja-JP pack").errorDescription?
            .contains("ja-JP") == true)
    }
}

/// Locale resolution for Apple's transcriber. These need the Speech assets, so they are
/// skipped where the framework cannot answer — but the fallback *rules* are checkable, and
/// they are what the en_JP bug was.
struct SystemTranscriberLocaleTests {
    @Test func anExactlySupportedLocaleIsUsedAsIs() async throws {
        let supported = await SystemTranscriber.supportedLocales
        try #require(!supported.isEmpty, "no speech assets on this machine")
        let first = supported[0]
        let resolved = await SystemTranscriber.resolvedLocale(for: first)
        #expect(resolved?.identifier(.bcp47) == first.identifier(.bcp47))
    }

    /// The bug. A language that is supported, in a region that is not, must resolve rather
    /// than fail — and must land on the language's likely region, not on whichever entry the
    /// list happened to start with.
    @Test func aSupportedLanguageInAnUnsupportedRegionResolves() async throws {
        let supported = await SystemTranscriber.supportedLocales
        try #require(supported.contains { $0.language.languageCode?.identifier == "en" },
                     "no English speech assets on this machine")
        let resolved = await SystemTranscriber.resolvedLocale(for: Locale(identifier: "en_JP"))
        #expect(resolved != nil, "English-in-Japan resolved to nothing")
        #expect(resolved?.language.languageCode?.identifier == "en")
        #expect(resolved?.region?.identifier == "US",
                "expected the language's likely region, got \(resolved?.identifier ?? "nil")")
    }

    @Test func aBareLanguageResolvesToItsLikelyRegion() async throws {
        let supported = await SystemTranscriber.supportedLocales
        try #require(supported.contains { $0.language.languageCode?.identifier == "en" })
        let resolved = await SystemTranscriber.resolvedLocale(for: Locale(identifier: "en"))
        #expect(resolved?.region?.identifier == "US")
    }

    /// A language Apple genuinely cannot do must still be refused. Resolving it to "some other
    /// language" would transcribe silently and wrongly, which is worse than an error.
    @Test func anUnsupportedLanguageIsRefused() async {
        let resolved = await SystemTranscriber.resolvedLocale(for: Locale(identifier: "tlh"))
        #expect(resolved == nil, "Klingon resolved to \(resolved?.identifier ?? "nil")")
    }
}
