// meeting-cli — the argument shell over `transcribeMeeting(audio:asr:diarizer:onTurn:)`
// (Sources/QuickStart.swift): parse flags, call the same function a GUI would ride, print the
// speaker-attributed transcript. Progress goes to stderr so stdout stays machine-checkable
// (agents: assert on stdout).
//
//   swift run meeting-cli --audio clip.wav
//   swift run meeting-cli --audio clip.wav --asr parakeet-tdt-0.6b-v3
//   swift run meeting-cli --audio clip.wav --diarizer nemotron-3-diarization --asr system
//
// --diarizer picks the diarization catalog id (sortformer-diar-v2, up to 4 speakers, or
// nemotron-3-diarization, up to 8). --asr system is Apple's transcriber (no download; --locale
// picks the language, default the Mac's). --diarizer-bundle composes a local diarizer bundle (a
// .aimodel/.aimodelc directory or its parent) with the ASR — the no-download door for
// conversion-tree gates.

import CoreAIKit
import Foundation

let usage = """
    usage: meeting-cli --audio <file> [--asr <catalog-id> | --asr system [--locale <id>]]
                       [--diarizer <catalog-id> | --diarizer-bundle <path>]
           meeting-cli --list-models
    (audio: anything AVFoundation reads — wav/m4a/mp3/…; resampled to 16 kHz mono)
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var audioPath: String?
var asrID = "whisper-large-v3-turbo"
var diarizerID = "sortformer-diar-v2"
var localeID: String?
var bundlePath: String?

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--audio": audioPath = args.popFirst()
    case "--asr": asrID = args.popFirst() ?? asrID
    case "--diarizer": diarizerID = args.popFirst() ?? diarizerID
    case "--locale": localeID = args.popFirst()
    case "--diarizer-bundle": bundlePath = args.popFirst()
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.diarization) {
            print("\(entry.id)  —  \(entry.name)  (diarization)")
        }
        for entry in ModelCatalog.builtin.available(.asr) {
            print("\(entry.id)  —  \(entry.name)  (asr)")
        }
        exit(0)
    case "--help", "-h":
        print(usage)
        exit(0)
    default:
        fail("unknown argument: \(arg)\n\(usage)")
    }
}

guard let audioPath else { fail(usage) }
let audioURL = URL(fileURLWithPath: (audioPath as NSString).expandingTildeInPath)

do {
    let started = ContinuousClock.now
    let transcript: MeetingTranscript
    if let bundlePath {
        // Local-bundle door: compose an on-disk diarizer with the ASR.
        let bundleURL = URL(fileURLWithPath: (bundlePath as NSString).expandingTildeInPath)
        stderrPrint("Loading diarizer from \(bundleURL.path) + ASR '\(asrID)'…")
        let transcriber: any SpeechToText = asrID == "system"
            ? try await SystemTranscriber(locale: localeID.map(Locale.init(identifier:)) ?? .current)
            : try await KitTranscriber(catalog: asrID)
        let meeting = MeetingTranscriber(
            diarizer: try await KitDiarizer(bundleAt: bundleURL), transcriber: transcriber)
        transcript = try await meeting.transcribe(
            samples: try AudioFile.pcm16kMono(audioURL)
        ) { stderrPrint("  " + $0.line) }
    } else if asrID == "system" {
        // The kit's default pairing: Apple transcribes, the diarizer is the only download.
        stderrPrint("Loading '\(diarizerID)' + Apple's transcriber (first run downloads the diarizer)…")
        let meeting = try await MeetingTranscriber(
            locale: localeID.map(Locale.init(identifier:)) ?? .current, diarizer: diarizerID)
        transcript = try await meeting.transcribe(
            samples: try AudioFile.pcm16kMono(audioURL)
        ) { stderrPrint("  " + $0.line) }
    } else {
        stderrPrint("Loading '\(diarizerID)' + ASR '\(asrID)' (first run downloads)…")
        transcript = try await transcribeMeeting(audio: audioURL, asr: asrID, diarizer: diarizerID) {
            stderrPrint("  " + $0.line)
        }
    }
    let elapsed = ContinuousClock.now - started
    let wall = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
    stderrPrint("")
    print(transcript.text)
    stderrPrint(
        "\n\(transcript.speakerCount) speaker(s), \(transcript.turns.count) transcribed turn(s), "
            + String(format: "%.2f s wall (load + diarize + ASR).", wall))
} catch {
    fail("meeting-cli failed: \(error.localizedDescription)")
}
