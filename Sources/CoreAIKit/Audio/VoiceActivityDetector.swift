// VoiceActivityDetector.swift — where speech starts and stops.
//
// The speech design calls this "the only genuinely new component" behind `listen()`, and it
// is: everything else in the live speech path is composition of models that already exist.
// Finalising an utterance needs somebody to decide that a pause was the end of one, and no
// model does that — an ASR model transcribes a window it is handed, and choosing the window
// is this file's job.
//
// Apple's nearest offering is `SNClassifySoundRequest`, which can tell you a window contained
// speech. That is a classifier, not an endpointer: it works on roughly second-long windows and
// answers "was there speech", where an endpointer has to answer "has this person stopped
// talking" within a couple of hundred milliseconds or the app feels broken.
//
// ```swift
// var vad = VoiceActivityDetector()
// for event in vad.process(samples, at: elapsed) {
//     if case .speechEnded(let segment) = event {
//         let text = try await transcriber.transcribe(samples: clip(segment))
//     }
// }
// ```
//
// Also useful without a microphone anywhere in sight: `segments(in:)` splits a two-hour
// recording into utterances, which is how you feed a model whose window is thirty seconds.
//
// WHAT IT IS AND IS NOT
// ---------------------
// Energy against an adaptive noise floor, with hysteresis and a hangover. No model, no
// dependency, no training data. What that buys and what it does not:
//
//   * A fixed dB threshold is the obvious implementation and it is wrong. The level that
//     works in a quiet room clips the first syllable of every word in a café, and the level
//     that works in a café is deaf to a whisper. So the floor is tracked, and the thresholds
//     are relative to it.
//   * It cannot separate speech from other sound at the same level. A fan, a television, a
//     road: all speech as far as this is concerned. That is the honest limit of an energy
//     detector, and the fix is a model, not a better threshold.

import Foundation

/// One stretch of speech.
public struct SpeechSegment: Sendable, Equatable {
    /// Seconds from the start of the stream, on the clock the caller passed in.
    public let start: TimeInterval
    public let end: TimeInterval

    public var duration: TimeInterval { end - start }

    public init(start: TimeInterval, end: TimeInterval) {
        self.start = start
        self.end = end
    }
}

/// What the detector decided, as it decides it.
public enum VoiceEvent: Sendable, Equatable {
    /// Speech began. Emitted as soon as the onset is confirmed, so a UI can show that it is
    /// listening; the segment is not known yet.
    case speechStarted(at: TimeInterval)
    /// Speech ended, and this is the whole utterance — the unit to transcribe.
    case speechEnded(SpeechSegment)
}

/// Decides where speech starts and stops in a stream of samples.
///
/// Stateful and order-dependent, like `KitTracker`: it is a running estimate of the room and
/// of whether someone is currently talking. Feeding it out-of-order buffers produces
/// nonsense, so own it from one place.
public final class VoiceActivityDetector {
    public struct Options: Sendable {
        /// Analysis frame, in seconds. 20 ms is the usual compromise: short enough to place a
        /// boundary precisely, long enough that one glottal pulse does not swing the energy.
        public var frameDuration: TimeInterval
        /// How far above the noise floor, in dB, energy must rise to begin speech.
        public var enterAboveFloorDB: Float
        /// How far above the floor it must stay to remain speech. Lower than
        /// `enterAboveFloorDB` on purpose — with one threshold the decision chatters on every
        /// frame that sits on it, and the chatter becomes fragmented utterances.
        public var exitAboveFloorDB: Float
        /// Speech is held on for this long after energy drops. Stop consonants contain real
        /// silence — "stop" has a gap in the middle — and without a hangover a word becomes
        /// two utterances.
        public var hangover: TimeInterval
        /// Silence needed to end an utterance. **This is the latency knob.** Short and the app
        /// answers before the speaker has finished the sentence; long and it feels asleep.
        /// There is no correct value, only a choice, which is why it is here and not a
        /// constant.
        public var minimumSilence: TimeInterval
        /// Utterances shorter than this are discarded: a door, a click, a cough. Measured on
        /// the speech itself, not on the padded clip — padding is for the model's benefit and
        /// must not be able to promote a 40 ms click over the floor.
        public var minimumSpeech: TimeInterval
        /// An utterance is closed and a new one begun after this long, even mid-word.
        ///
        /// Not a nicety: an energy detector cannot tell steady noise from steady speech, so
        /// without an upper bound a fan that switches on holds one utterance open forever, and
        /// what finally reaches the model is a buffer longer than its window. Whisper's is
        /// 30 s, so the default sits well under it. Splitting mid-sentence is the lesser
        /// failure — the alternative is a clip the model cannot take at all.
        public var maximumSpeech: TimeInterval
        /// How fast the noise floor follows a room that got louder. Slow, because speech must
        /// not be able to drag the floor up to meet it — that is how a detector goes deaf
        /// during a long sentence.
        public var floorRiseHalfLife: TimeInterval
        /// How fast it follows a room that got quieter. Fast: when a noise stops, the
        /// detector should be able to hear again immediately.
        public var floorFallHalfLife: TimeInterval

        public init(
            frameDuration: TimeInterval = 0.02,
            enterAboveFloorDB: Float = 9,
            exitAboveFloorDB: Float = 5,
            hangover: TimeInterval = 0.2,
            minimumSilence: TimeInterval = 0.5,
            minimumSpeech: TimeInterval = 0.15,
            maximumSpeech: TimeInterval = 15,
            floorRiseHalfLife: TimeInterval = 3.0,
            floorFallHalfLife: TimeInterval = 0.3
        ) {
            self.maximumSpeech = maximumSpeech
            self.frameDuration = frameDuration
            self.enterAboveFloorDB = enterAboveFloorDB
            self.exitAboveFloorDB = exitAboveFloorDB
            self.hangover = hangover
            self.minimumSilence = minimumSilence
            self.minimumSpeech = minimumSpeech
            self.floorRiseHalfLife = floorRiseHalfLife
            self.floorFallHalfLife = floorFallHalfLife
        }

        /// Answers sooner, at the cost of cutting in on someone who pauses to think. For a
        /// command interface, where utterances are short and the wait is the whole experience.
        public static let responsive = Options(hangover: 0.15, minimumSilence: 0.28)

        /// Waits longer before deciding a sentence ended. For dictation and meetings, where
        /// splitting a sentence in half costs more than a moment of latency.
        public static let patient = Options(hangover: 0.35, minimumSilence: 0.9)
    }

    public let options: Options
    private let sampleRate: Double
    private let frameSamples: Int

    /// Noise floor in dBFS. Starts pessimistically low so the first loud thing is heard; the
    /// tracker pulls it up to the real room within `floorRiseHalfLife`.
    private var floorDB: Float = -70
    private var hasFloor = false

    private var carry: [Float] = []
    private var elapsed: TimeInterval = 0

    private var speechStart: TimeInterval?
    /// First frame that actually crossed the threshold. `speechStart` is back-dated by a
    /// hangover for the model's benefit; the minimum-length decision is made on this.
    private var firstLoud: TimeInterval?
    private var lastLoud: TimeInterval?
    private var silenceStart: TimeInterval?

    public init(sampleRate: Double = 16000, options: Options = Options()) {
        self.options = options
        self.sampleRate = sampleRate
        self.frameSamples = max(1, Int(sampleRate * options.frameDuration))
    }

    /// Current background level in dBFS, for a meter or for deciding the mic is dead.
    public var noiseFloorDB: Float { floorDB }

    /// Whether the detector currently believes someone is talking.
    public var isSpeaking: Bool { speechStart != nil }

    public func reset() {
        carry.removeAll()
        elapsed = 0
        floorDB = -70
        hasFloor = false
        speechStart = nil
        firstLoud = nil
        lastLoud = nil
        silenceStart = nil
    }

    /// Feed the next buffer. Returns whatever it decided while consuming it.
    ///
    /// Buffers of any length are fine — samples that do not fill a frame are carried into the
    /// next call, so a 512-sample tap and a 16000-sample chunk give the same answers.
    @discardableResult
    public func process(_ samples: [Float]) -> [VoiceEvent] {
        var events: [VoiceEvent] = []
        carry.append(contentsOf: samples)

        var offset = 0
        while offset + frameSamples <= carry.count {
            let frame = carry[offset..<(offset + frameSamples)]
            offset += frameSamples
            let frameEnd = elapsed + options.frameDuration
            events.append(contentsOf: consume(frame, endingAt: frameEnd))
            elapsed = frameEnd
        }
        carry.removeFirst(offset)
        return events
    }

    /// Flushes the tail: if speech was in progress when the stream ended, close it. Otherwise
    /// the last utterance of every recording is silently lost, which is the kind of bug that
    /// only shows up on the one sentence that mattered.
    public func finish() -> [VoiceEvent] {
        defer { speechStart = nil; firstLoud = nil; lastLoud = nil; silenceStart = nil }
        guard let start = speechStart, let first = firstLoud, let last = lastLoud else {
            return []
        }
        return emit(start: start, firstLoud: first, end: last)
    }

    // MARK: - One frame

    private func consume(_ frame: ArraySlice<Float>, endingAt time: TimeInterval) -> [VoiceEvent] {
        let db = VoiceActivityDetector.decibels(frame)

        // The floor follows quiet quickly and loud slowly, so that speech cannot drag it up to
        // meet itself. Seeded from the first frame rather than converged to over seconds — a
        // recording that opens mid-sentence should not spend three seconds deaf.
        if !hasFloor {
            floorDB = db
            hasFloor = true
        } else {
            let halfLife = db < floorDB ? options.floorFallHalfLife : options.floorRiseHalfLife
            let alpha = Float(1 - pow(0.5, options.frameDuration / max(halfLife, 1e-6)))
            floorDB += (db - floorDB) * alpha
        }

        let threshold = floorDB + (isSpeaking ? options.exitAboveFloorDB : options.enterAboveFloorDB)
        let loud = db >= threshold

        if loud {
            let previousLoud = lastLoud
            lastLoud = time
            silenceStart = nil
            // Bounded, even while the speaker (or the fan) carries on. Closed here and
            // reopened immediately, so nothing is dropped — only cut.
            if let start = speechStart, let first = firstLoud,
                time - start >= options.maximumSpeech
            {
                let closed = emit(start: start, firstLoud: first, end: previousLoud ?? time)
                speechStart = time - options.frameDuration
                firstLoud = time
                return closed + [.speechStarted(at: time - options.frameDuration)]
            }
            if speechStart == nil {
                firstLoud = time
                // Back-date the onset by one hangover: the frame that crossed the threshold is
                // not the first frame of the word, it is the first frame loud enough to be
                // sure. Starting the clip here clips the consonant off the front.
                let start = max(0, time - options.frameDuration - options.hangover)
                speechStart = start
                return [.speechStarted(at: start)]
            }
            return []
        }

        guard let start = speechStart, let first = firstLoud, let last = lastLoud else {
            return []
        }
        // Quiet, but speech is open. The hangover keeps it open across a stop consonant; only
        // after that does silence start counting toward the end of the utterance.
        if time - last <= options.hangover { return [] }
        let silenceBegan = silenceStart ?? last
        silenceStart = silenceBegan
        guard time - silenceBegan >= options.minimumSilence else { return [] }

        speechStart = nil
        firstLoud = nil
        lastLoud = nil
        silenceStart = nil
        return emit(start: start, firstLoud: first, end: last)
    }

    private func emit(
        start: TimeInterval, firstLoud: TimeInterval, end: TimeInterval
    ) -> [VoiceEvent] {
        // Length is judged on the measured speech; the clip handed on is padded by a hangover
        // at each end, because a word's first consonant and last breath sit below threshold
        // and an ASR model does better with them than without.
        guard end - firstLoud >= options.minimumSpeech else { return [] }
        return [.speechEnded(SpeechSegment(start: start, end: end + options.hangover))]
    }

    /// RMS in dBFS. Floored well below anything a real microphone produces so that digital
    /// silence is a number rather than negative infinity.
    static func decibels(_ frame: ArraySlice<Float>) -> Float {
        var sum: Float = 0
        for sample in frame { sum += sample * sample }
        let rms = (sum / Float(frame.count)).squareRoot()
        return rms > 1e-9 ? 20 * log10(rms) : -120
    }
}

extension VoiceActivityDetector {
    /// Splits a whole waveform into utterances, in one call.
    ///
    /// This is the non-live half and it is immediately useful on its own: an ASR model takes a
    /// window of tens of seconds, a recording is hours, and cutting at pauses rather than at a
    /// fixed stride is the difference between transcribing sentences and transcribing halves
    /// of words.
    public static func segments(
        in samples: [Float], sampleRate: Double = 16000, options: Options = Options()
    ) -> [SpeechSegment] {
        let detector = VoiceActivityDetector(sampleRate: sampleRate, options: options)
        var found: [SpeechSegment] = []
        for event in detector.process(samples) + detector.finish() {
            if case .speechEnded(let segment) = event { found.append(segment) }
        }
        return found
    }
}

extension SpeechSegment {
    /// The samples this segment covers, clamped to the waveform.
    public func samples(from waveform: [Float], sampleRate: Double = 16000) -> [Float] {
        let from = max(0, Int(start * sampleRate))
        let to = min(waveform.count, Int(end * sampleRate))
        guard from < to else { return [] }
        return Array(waveform[from..<to])
    }
}
