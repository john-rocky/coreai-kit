// meeting-cli — the argument shell over `transcribeMeeting(audio:asr:onTurn:)`
// (Sources/QuickStart.swift): parse flags, call the same function a GUI would ride, print the
// speaker-attributed transcript. Progress goes to stderr so stdout stays machine-checkable
// (agents: assert on stdout).
//
//   swift run meeting-cli --audio clip.wav
//   swift run meeting-cli --audio clip.wav --asr parakeet-tdt-0.6b-v3
//
// --diarizer-bundle composes a local Sortformer bundle (a .aimodel/.aimodelc directory or its
// parent) with the catalog ASR — the no-download door for conversion-tree gates.

import CoreAIKit
import Foundation

let usage = """
    usage: meeting-cli --audio <file> [--asr <catalog-id>] [--diarizer-bundle <path>]
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
var bundlePath: String?

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--audio": audioPath = args.popFirst()
    case "--asr": asrID = args.popFirst() ?? asrID
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
    let transcript: MeetingTranscript
    if let bundlePath {
        // Local-bundle door: compose an on-disk diarizer with the catalog ASR.
        let bundleURL = URL(fileURLWithPath: (bundlePath as NSString).expandingTildeInPath)
        stderrPrint("Loading diarizer from \(bundleURL.path) + ASR '\(asrID)'…")
        let meeting = MeetingTranscriber(
            diarizer: try await KitDiarizer(bundleAt: bundleURL),
            transcriber: try await KitTranscriber(catalog: asrID))
        transcript = try await meeting.transcribe(
            samples: try AudioFile.pcm16kMono(audioURL)
        ) { stderrPrint("  " + $0.line) }
    } else {
        stderrPrint("Loading 'sortformer-diar-v2' + ASR '\(asrID)' (first run downloads)…")
        transcript = try await transcribeMeeting(audio: audioURL, asr: asrID) {
            stderrPrint("  " + $0.line)
        }
    }
    stderrPrint("")
    print(transcript.text)
    stderrPrint(
        "\n\(transcript.speakerCount) speaker(s), \(transcript.turns.count) transcribed turn(s).")
} catch {
    fail("meeting-cli failed: \(error.localizedDescription)")
}
