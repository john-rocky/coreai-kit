// SystemTranscriber.swift — Apple's speech-to-text, behind this package's ASR protocol.
//
// The catalog's ASR models cost between 1.3 GB and 3.2 GB on an iPhone. iOS 27 ships
// `SpeechAnalyzer` + `SpeechTranscriber`, which cost **zero bytes of app download**: the
// locale assets are OS-managed and shared with every other app on the device. Streaming,
// word timestamps and a per-locale asset inventory are all in there.
//
// This package's own porting contract has a gate for exactly this — "Apple's stock stack does
// not already ship this capability. If it does, stop." — and speech-to-text is the clearest
// case of it in the whole catalog. So the default transcription path is Apple's, and a
// gigabyte-class catalog model is what you opt *into*, for the reasons that survive the gate:
//
//   * a locale Apple does not support on the device in hand (`isAvailable(for:)` answers this),
//   * a model whose behaviour must not change when the OS updates,
//   * an offline guarantee on a device whose assets were never installed.
//
// What Apple still does not ship, and what this kit is therefore actually for in speech, is
// **who spoke when**. `MeetingTranscriber` composes this transcriber with `KitDiarizer`, and
// that pairing is 238 MB rather than 3.4 GB.
//
// ```swift
// let asr = try await SystemTranscriber()                  // 0 bytes
// let meeting = try await MeetingTranscriber(asr: asr)     // + Sortformer, 238 MB
// ```

import AVFoundation
import Foundation
import Speech

/// Anything that turns a 16 kHz mono waveform into text.
///
/// The seam that lets the expensive model be optional: `MeetingTranscriber` and the ops take
/// this rather than a concrete class, so Apple's free transcriber and a catalog model are
/// interchangeable at the call site.
public protocol SpeechToText: Sendable {
    /// What produced the text, for a transcript header or a bug report.
    var id: String { get }

    func transcribe(
        samples: [Float], language: String?, onPartial: (@Sendable (String) -> Void)?
    ) async throws -> Transcription
}

extension SpeechToText {
    public func transcribe(samples: [Float], language: String? = nil) async throws -> Transcription
    {
        try await transcribe(samples: samples, language: language, onPartial: nil)
    }
}

extension KitTranscriber: SpeechToText {}

/// Apple's on-device speech-to-text (`SpeechAnalyzer` + `SpeechTranscriber`), at no download
/// cost to the app.
public struct SystemTranscriber: SpeechToText {
    public enum TranscriberError: LocalizedError {
        case localeUnsupported(Locale)
        case conversionFailed

        public var errorDescription: String? {
            switch self {
            case .localeUnsupported(let locale):
                return "The system transcriber does not support \(locale.identifier) on this "
                    + "device. `SystemTranscriber.supportedLocales` lists what it does."
            case .conversionFailed:
                return "Could not convert the waveform to the analyzer's audio format."
            }
        }
    }

    public let locale: Locale
    public var id: String { "system-speech(\(locale.identifier))" }

    /// Locales Apple can transcribe on this device, whether or not the assets are installed.
    public static var supportedLocales: [Locale] {
        get async { await SpeechTranscriber.supportedLocales }
    }

    /// Locales whose assets are already on the device — these cost nothing and no wait.
    public static var installedLocales: [Locale] {
        get async { await SpeechTranscriber.installedLocales }
    }

    /// The locale Apple will actually transcribe in, for a requested one — or nil if it
    /// cannot do that language at all.
    ///
    /// **Exact matching is wrong here and the bug it causes is invisible.** A device set to
    /// English-in-Japan reports `en_JP`, which is in nobody's supported list, while `en_US`,
    /// `en_GB` and eight other English locales are installed and working. Matching strictly
    /// would tell that user transcription is unsupported on a device where it plainly works —
    /// and nobody files a report about a feature they never saw. (Found the first time this
    /// ran on a real machine, which was not the developer's own default locale.)
    ///
    /// So: the exact locale if it is offered, else the same language in the requested region,
    /// else the same language anywhere — preferring one already installed, since that costs
    /// no download and no wait.
    public static func resolvedLocale(for requested: Locale = .current) async -> Locale? {
        let supported = await SpeechTranscriber.supportedLocales
        let installed = Set(await SpeechTranscriber.installedLocales.map { $0.identifier(.bcp47) })
        let wanted = requested.identifier(.bcp47)
        if let exact = supported.first(where: { $0.identifier(.bcp47) == wanted }) {
            return exact
        }
        guard let language = requested.language.languageCode?.identifier else { return nil }
        let sameLanguage = supported.filter {
            $0.language.languageCode?.identifier == language
        }
        guard !sameLanguage.isEmpty else { return nil }
        // Requested region first — an Australian asking for English should get en_AU. Then
        // the language's *likely* region, which is what ICU's maximal identifier is for:
        // "en" maximises to "en_Latn_US", so English-in-Japan lands on en_US rather than on
        // whichever English locale happened to be first in the installed list.
        let region = requested.region?.identifier
        // `maximalIdentifier` is hyphenated — "en" gives "en-Latn-US" — while the locale
        // identifiers beside it are underscored. Splitting on one separator silently matches
        // nothing, which is a fallback that looks implemented and is not.
        let likely = Locale.Language(identifier: language).maximalIdentifier
            .split(whereSeparator: { $0 == "-" || $0 == "_" }).last.map(String.init)
        return sameLanguage.first { $0.region?.identifier == region }
            ?? sameLanguage.first { $0.region?.identifier == likely }
            ?? sameLanguage.first { installed.contains($0.identifier(.bcp47)) }
            ?? sameLanguage.first
    }

    /// Whether this locale's assets are already on the device — no wait, no network.
    /// `isAvailable` says the OS *can* do this language; this says it can do it right now.
    public static func isInstalled(for locale: Locale = .current) async -> Bool {
        guard let resolved = await resolvedLocale(for: locale) else { return false }
        let wanted = resolved.identifier(.bcp47)
        return await SpeechTranscriber.installedLocales
            .contains { $0.identifier(.bcp47) == wanted }
    }

    public static func isAvailable(for locale: Locale = .current) async -> Bool {
        await resolvedLocale(for: locale) != nil
    }

    /// Prepares the transcriber, downloading the locale's assets if the OS does not have them
    /// yet. The download is Apple's, not the app's: it is shared with every other app and does
    /// not count against the app's size.
    public init(locale requested: Locale = .current) async throws {
        guard let locale = await SystemTranscriber.resolvedLocale(for: requested) else {
            throw TranscriberError.localeUnsupported(requested)
        }
        self.locale = locale
        let module = SpeechTranscriber(locale: locale, preset: .transcription)
        if await AssetInventory.status(forModules: [module]) != .installed,
            let request = try await AssetInventory.assetInstallationRequest(
                supporting: [module])
        {
            try await request.downloadAndInstall()
        }
    }

    /// Transcribe a 16 kHz mono waveform.
    ///
    /// `language` is ignored: the locale is fixed at init because Apple's transcriber is
    /// per-locale by construction, and silently transcribing in the wrong one is worse than
    /// saying so. Construct another `SystemTranscriber` for another language.
    public func transcribe(
        samples: [Float], language: String? = nil,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        guard !samples.isEmpty else { return Transcription(language: locale.identifier, text: "") }

        let module = SpeechTranscriber(locale: locale, preset: .transcription)
        let analyzer = SpeechAnalyzer(modules: [module])
        let inputs = try await SystemTranscriber.analyzerInputs(
            for: samples, modules: [module])

        // Results are consumed concurrently with the analysis: the sequence finishes when the
        // analyzer finalises, so collecting after would deadlock on a stream nobody is draining.
        async let collected: String = {
            var text = ""
            for try await result in module.results where result.isFinal {
                text += String(result.text.characters)
                if let onPartial { onPartial(text) }
            }
            return text
        }()

        let stream = AsyncStream<AnalyzerInput> { continuation in
            for input in inputs { continuation.yield(input) }
            continuation.finish()
        }
        _ = try await analyzer.analyzeSequence(stream)
        try await analyzer.finalizeAndFinishThroughEndOfInput()

        let text = try await collected
        return Transcription(
            language: locale.identifier,
            text: text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Converts the waveform into whatever format the analyzer asked for.
    ///
    /// The kit speaks 16 kHz mono Float everywhere because that is what the catalog models
    /// take. Apple's analyzer names its own preferred format and **traps** on a buffer in a
    /// different one — not an error, a crash — so `AnalyzerInputConverter` is not optional
    /// politeness. (Written the naive way first, which is how that was discovered.)
    private static func analyzerInputs(
        for samples: [Float], modules: [any SpeechModule]
    ) async throws -> [AnalyzerInput] {
        guard
            let source = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1,
                interleaved: false),
            let buffer = AVAudioPCMBuffer(
                pcmFormat: source, frameCapacity: AVAudioFrameCount(samples.count))
        else { throw TranscriberError.conversionFailed }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channel = buffer.floatChannelData?[0] else {
            throw TranscriberError.conversionFailed
        }
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: $0.count) }

        let converter = try await AnalyzerInputConverter.converter(compatibleWith: modules)
        return try converter.convert(buffer, at: nil) + converter.flush()
    }
}
