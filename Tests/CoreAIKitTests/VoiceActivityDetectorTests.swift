// VoiceActivityDetectorTests.swift — the decisions, on signals whose answer is known.
//
// A VAD is easy to write and hard to know you got right, because on real speech every output
// looks plausible. So the inputs here are constructed: a burst of known length at a known
// level in known noise, where the right answer is arithmetic rather than opinion. The cases
// are the ones that break energy detectors in the field — a stop consonant mid-word, a room
// that gets louder, a long sentence, a click.

import Foundation
import Testing

@testable import CoreAIKit

struct VoiceActivityDetectorTests {
    private let rate = 16000.0

    /// Deterministic pseudo-noise: a real `random` makes a failure impossible to reproduce.
    private func noise(_ seconds: Double, level: Float, seed: UInt64 = 1) -> [Float] {
        var state = seed &* 6_364_136_223_846_793_005 &+ 1
        return (0..<Int(seconds * rate)).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return (Float(state >> 40) / Float(1 << 23) - 1) * level
        }
    }

    /// A tone at a level, standing in for a voiced sound.
    private func tone(_ seconds: Double, level: Float, hz: Double = 220) -> [Float] {
        (0..<Int(seconds * rate)).map { i in
            level * Float(sin(2 * Double.pi * hz * Double(i) / rate))
        }
    }

    private func segments(_ waveform: [Float], _ options: VoiceActivityDetector.Options = .init())
        -> [SpeechSegment]
    {
        VoiceActivityDetector.segments(in: waveform, sampleRate: rate, options: options)
    }

    // MARK: - The basics

    @Test func silenceIsNotSpeech() {
        #expect(segments(noise(3, level: 0.001)).isEmpty)
    }

    @Test func digitalSilenceDoesNotProduceInfinities() {
        let detector = VoiceActivityDetector(sampleRate: rate)
        _ = detector.process([Float](repeating: 0, count: 16000))
        #expect(detector.noiseFloorDB.isFinite)
        #expect(!detector.isSpeaking)
    }

    @Test func oneBurstIsOneSegment() {
        let waveform = noise(1, level: 0.002) + tone(0.8, level: 0.2) + noise(1.5, level: 0.002)
        let found = segments(waveform)

        #expect(found.count == 1)
        guard let s = found.first else { return }
        // Boundaries within a hangover of the truth on each side — the detector deliberately
        // widens the clip rather than cutting a consonant off.
        #expect(abs(s.start - 1.0) < 0.3, "start \(s.start)")
        #expect(abs(s.end - 1.8) < 0.3, "end \(s.end)")
    }

    @Test func twoBurstsFarApartAreTwoSegments() {
        let waveform = noise(0.5, level: 0.002)
            + tone(0.5, level: 0.2) + noise(1.5, level: 0.002)
            + tone(0.5, level: 0.2) + noise(1.0, level: 0.002)
        #expect(segments(waveform).count == 2)
    }

    /// The case a naive detector always gets wrong. "Stop" contains real silence in the middle
    /// of the word; without a hangover it becomes two utterances and the transcript reads as
    /// two fragments.
    @Test func aStopConsonantDoesNotSplitAWord() {
        let waveform = noise(0.5, level: 0.002)
            + tone(0.3, level: 0.2)
            + noise(0.08, level: 0.002)  // the closure, 80 ms
            + tone(0.3, level: 0.2)
            + noise(1.5, level: 0.002)
        let found = segments(waveform)

        #expect(found.count == 1, "an 80 ms gap split the word into \(found.count) pieces")
        #expect((found.first?.duration ?? 0) > 0.6)
    }

    @Test func aClickIsRejected() {
        // 40 ms, under the 150 ms floor: a door, not a word.
        let waveform = noise(0.5, level: 0.002) + tone(0.04, level: 0.3)
            + noise(1.5, level: 0.002)
        #expect(segments(waveform).isEmpty)
    }

    // MARK: - The room

    /// A fixed threshold is the obvious implementation and this is why it is wrong: the same
    /// level of speech has to be found in a quiet room and in a loud one.
    @Test func theSameSpeechIsFoundAtAnyBackgroundLevel() {
        for floor in [Float(0.0005), 0.002, 0.01, 0.03] {
            let waveform = noise(2, level: floor)
                + tone(0.6, level: floor * 12)
                + noise(1.5, level: floor)
            let found = segments(waveform)
            #expect(found.count == 1, "background \(floor): found \(found.count) segments")
        }
    }

    /// A fan switching on is a step in level, and an energy detector cannot tell that from
    /// someone starting to talk — it will fire, and the honest question is what happens next.
    /// Two things must: the floor climbs to meet the noise, and the utterance is bounded.
    ///
    /// The first version of this test asserted the noise was reported as under two seconds of
    /// speech. The arithmetic says that is not deliverable: absorbing a 34 dB step through a
    /// 3 s half-life takes about 13 s to come within the 5 dB exit margin, and shortening the
    /// half-life to hit two seconds is exactly what makes a long sentence go deaf half way
    /// through (the test below). So the assertion was wrong, not the detector — but the
    /// unbounded utterance it exposed was real, and `maximumSpeech` came from it.
    @Test func steadyNoiseIsAbsorbedAndBounded() {
        let detector = VoiceActivityDetector(
            sampleRate: rate, options: .init(maximumSpeech: 4))
        var found: [SpeechSegment] = []
        for event in detector.process(noise(1, level: 0.001) + noise(20, level: 0.05, seed: 7))
            + detector.finish()
        {
            if case .speechEnded(let s) = event { found.append(s) }
        }

        // Bounded: no clip longer than the model's window, whatever the room does.
        #expect(found.allSatisfy { $0.duration < 4.5 },
                "longest \(found.map(\.duration).max() ?? 0)s")
        // Absorbed: the floor has climbed to the noise, so it is no longer being heard as an
        // onset. -26 dBFS is the level of the noise itself.
        #expect(detector.noiseFloorDB > -32, "floor stuck at \(detector.noiseFloorDB) dBFS")
        #expect(!detector.isSpeaking, "still hearing speech after 20 s of constant noise")
    }

    /// The mirror of the last one, and the reason the floor falls faster than it rises: speech
    /// must not be able to raise the floor to its own level and go deaf part way through.
    @Test func aLongUtteranceDoesNotGoDeafAtTheEnd() {
        let waveform = noise(1, level: 0.002) + tone(6, level: 0.2) + noise(1.5, level: 0.002)
        let found = segments(waveform)

        #expect(found.count == 1, "a 6 s utterance came out in \(found.count) pieces")
        #expect((found.first?.duration ?? 0) > 5.0, "got \(found.first?.duration ?? 0)s of 6")
    }

    // MARK: - Streaming

    /// A microphone tap hands over whatever buffer size it feels like. The answers must not
    /// depend on it.
    @Test func bufferSizeDoesNotChangeTheAnswer() {
        let waveform = noise(0.5, level: 0.002) + tone(0.6, level: 0.2)
            + noise(1.5, level: 0.002)
        var byChunk: [[SpeechSegment]] = []
        for size in [160, 512, 4096, waveform.count] {
            let detector = VoiceActivityDetector(sampleRate: rate)
            var found: [SpeechSegment] = []
            var offset = 0
            while offset < waveform.count {
                let end = min(offset + size, waveform.count)
                for event in detector.process(Array(waveform[offset..<end])) {
                    if case .speechEnded(let s) = event { found.append(s) }
                }
                offset = end
            }
            for event in detector.finish() {
                if case .speechEnded(let s) = event { found.append(s) }
            }
            byChunk.append(found)
        }
        #expect(byChunk.allSatisfy { $0.count == byChunk[0].count })
        for run in byChunk.dropFirst() {
            #expect(abs(run[0].start - byChunk[0][0].start) < 0.05)
            #expect(abs(run[0].end - byChunk[0][0].end) < 0.05)
        }
    }

    @Test func startIsAnnouncedBeforeTheUtteranceEnds() {
        let detector = VoiceActivityDetector(sampleRate: rate)
        var events = detector.process(noise(0.5, level: 0.002) + tone(0.4, level: 0.2))
        #expect(events.contains { if case .speechStarted = $0 { true } else { false } })
        #expect(!events.contains { if case .speechEnded = $0 { true } else { false } },
                "the end was announced while the speaker was still talking")
        events = detector.process(noise(1.5, level: 0.002))
        #expect(events.contains { if case .speechEnded = $0 { true } else { false } })
    }

    /// Speech still open when the stream ends. Without a flush the last utterance of every
    /// recording disappears, which is the failure nobody notices until it is the one that
    /// mattered.
    @Test func theTailIsNotLost() {
        let detector = VoiceActivityDetector(sampleRate: rate)
        _ = detector.process(noise(0.5, level: 0.002) + tone(0.6, level: 0.2))
        let tail = detector.finish()
        #expect(tail.count == 1)
        if case .speechEnded(let s) = tail.first { #expect(s.duration > 0.4) } else {
            Issue.record("finish() did not close the open utterance")
        }
    }

    // MARK: - The knob

    /// `minimumSilence` is the latency choice the design says must be exposed rather than
    /// guessed. Both presets must be self-consistent: the responsive one cuts at a pause the
    /// patient one rides through.
    @Test func thePresetsDifferInTheWayTheyClaimTo() {
        let waveform = noise(0.5, level: 0.002)
            + tone(0.4, level: 0.2)
            + noise(0.45, level: 0.002)  // a pause to think
            + tone(0.4, level: 0.2)
            + noise(1.5, level: 0.002)

        #expect(segments(waveform, .responsive).count == 2, "responsive should cut at the pause")
        #expect(segments(waveform, .patient).count == 1, "patient should ride through it")
    }

    @Test func segmentSamplesAreClampedToTheWaveform() {
        let waveform = tone(1, level: 0.2)
        let past = SpeechSegment(start: 0.5, end: 99)
        #expect(past.samples(from: waveform, sampleRate: rate).count == 8000)
        let before = SpeechSegment(start: -5, end: 0.1)
        #expect(before.samples(from: waveform, sampleRate: rate).count == 1600)
        #expect(SpeechSegment(start: 5, end: 1).samples(from: waveform, sampleRate: rate).isEmpty)
    }
}
