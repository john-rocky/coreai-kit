// KVCacheEndTests.swift — the stops at the end of a fixed KV cache, without a model: the position
// check every `StatefulGraphModel.step` runs, the frame budget VoxCPM and VoxCPM2 run under, the
// voice check VibeVoice runs before it speaks, and what their errors say. The published bundles of
// these three models hold 512 positions; VibeVoice's windows are 5 text tokens and 6 speech frames
// (`coreai_host/glue/glue.json`), and its voice prompts take up to 510 positions of the speech model.
//
//     swift test --filter KVCacheEnd

import Foundation
import Testing

@testable import CoreAIKit
@testable import CoreAIKitVision

struct KVCacheEndTests {
    // MARK: - StatefulGraphModel.step

    @available(macOS 27, iOS 27, *)
    static func rejectedPosition(_ inputs: [String: TensorValue], capacity: Int) -> Int? {
        do {
            try StatefulGraphModel.checkPosition(inputs, capacity: capacity)
            return nil
        } catch VisionError.kvCacheFull(let position, let cap) {
            return cap == capacity ? position : -1
        } catch {
            return -2
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func aStepInsideTheCacheRuns() {
        for p: Int32 in [0, 1, 255, 511] {
            #expect(Self.rejectedPosition(["pos": .int32([p], shape: [1])], capacity: 512) == nil)
        }
        // A batched prefill has no `pos`: it writes from position 0 and is bounded by its shape.
        #expect(Self.rejectedPosition(["inputs_embeds": .float32([0, 0], shape: [1, 2])], capacity: 512) == nil)
    }

    @available(macOS 27, iOS 27, *)
    @Test func aStepAtOrPastTheEndThrowsBeforeTheGraphRuns() {
        for p: Int32 in [512, 513, 1000] {
            #expect(Self.rejectedPosition(["pos": .int32([p], shape: [1])], capacity: 512) == Int(p))
        }
        #expect(Self.rejectedPosition(["pos": .int32([2048], shape: [1])], capacity: 2048) == 2048)
    }

    // MARK: - VoxCPM / VoxCPM2: frames

    @available(macOS 27, iOS 27, *)
    @Test func aTextThatLeavesRoomKeepsMaxFrames() throws {
        // The text takes T positions (its tokens plus <|audio_start|>); frame 0 comes from the
        // prefill and frame i + 1 from the decode at T + i, so 300 frames decode up to T + 298:
        // 212 + 298 = 510 and 213 + 298 = 511.
        #expect(try VoxCPMTTS.frameBudget(textPositions: 7, maxFrames: 300, capacity: 512) == 300)
        #expect(try VoxCPMTTS.frameBudget(textPositions: 212, maxFrames: 300, capacity: 512) == 300)
        #expect(try VoxCPMTTS.frameBudget(textPositions: 213, maxFrames: 300, capacity: 512) == 300)
        #expect(try VoxCPMTTS.frameBudget(textPositions: 400, maxFrames: 50, capacity: 512) == 50)
    }

    @available(macOS 27, iOS 27, *)
    @Test func aLongTextEndsAtTheLastPosition() throws {
        for t in [214, 300, 450, 511] {
            let frames = try VoxCPMTTS.frameBudget(textPositions: t, maxFrames: 300, capacity: 512)
            #expect(frames == 513 - t)
            #expect(t + frames - 2 == 511)  // the decode before the last frame
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func aTextThatFillsTheCacheThrows() {
        for t in [512, 513, 600] {
            let error = #expect(throws: VoxCPMError.self) {
                _ = try VoxCPMTTS.frameBudget(textPositions: t, maxFrames: 300, capacity: 512)
            }
            guard case .textTooLong(let positions, let capacity)? = error else {
                Issue.record("T = \(t): expected textTooLong, got \(String(describing: error))")
                continue
            }
            #expect(positions == t && capacity == 512)
        }
    }

    // MARK: - VibeVoice: voices

    @available(macOS 27, iOS 27, *)
    @Test func everyVoiceButTheLongestLeavesRoomToSpeak() {
        // tts_prefill_len in the published voices/index.json: en-Carter_man 316 (the longest en-*),
        // de-Spk1_woman 394, kr-Spk1_man 421, de-Spk0_man 510.
        for prompt in [0, 316, 394, 421, 501] {
            #expect(VibeVoiceTTS.leavesRoom(
                promptPositions: prompt, capacity: 512, textWindow: 5, speechWindow: 6))
        }
        for prompt in [502, 510, 512] {
            #expect(!VibeVoiceTTS.leavesRoom(
                promptPositions: prompt, capacity: 512, textWindow: 5, speechWindow: 6))
        }
    }

    // MARK: - The errors

    @Test func theErrorsNameTheLimitAndWhatToDo() throws {
        let text = try #require(VoxCPMError.textTooLong(positions: 600, capacity: 512).errorDescription)
        #expect(text.contains("600 of the 512 positions") && text.contains("one sentence at a time"))
        let voice = try #require(
            VibeVoiceError.voiceFillsCache(voice: "de-Spk0_man", promptPositions: 510, capacity: 512)
                .errorDescription)
        #expect(voice.contains("de-Spk0_man") && voice.contains("510 of the 512 positions"))
        #expect(VisionError.kvCacheFull(position: 512, capacity: 512).errorDescription?
            .contains("position 512") == true)
        // The cases VibeVoiceError had before keep the system's description.
        #expect(VibeVoiceError.emptyText.errorDescription == nil)
        #expect(VibeVoiceError.voiceNotFound("x").errorDescription == nil)
        #expect(VibeVoiceError.assetMissing("x").errorDescription == nil)
    }
}
