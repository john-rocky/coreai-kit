// AudioChatModel.swift — view model: loads a local Qwen2.5-Omni audio-understanding bundle as a
// KitAudioModel, attaches a chosen/recorded clip (mel → encoder → static buffer), and answers a
// question through a plain FoundationModels `LanguageModelSession`.

import CoreAIKit
import Foundation
import FoundationModels
import os

private let log = Logger(subsystem: "com.coreaikit.audiochat", category: "selftest")

/// MB of memory the app may still allocate before iOS jetsams it (iOS only; 0 elsewhere).
func availableMemoryMB() -> Int {
    #if os(iOS)
        return Int(os_proc_available_memory() / (1024 * 1024))
    #else
        return 0
    #endif
}

@MainActor
final class AudioChatModel: ObservableObject {
    @Published var status = "Tap “Load model”."
    @Published var loaded = false
    @Published var busy = false
    @Published var clipName = "No audio loaded."
    @Published var answer = ""

    // macOS: local bundles from the conversion repo (the GPU-pipelined .aimodel decoder).
    // iOS: the AOT decoder (.aimodelc, dodges the on-device JIT jetsam) + JIT encoder, sideloaded
    // into the app container's Documents/models via `devicectl device copy to`.
    #if os(macOS)
        private let decoderDir =
            "/Users/majimadaisuke/code/coreai/ondevice/artifacts/qwen2_5_omni_3b_thinker_int8lin_n750_s1_bundle"
        private let encoderModel =
            "/Users/majimadaisuke/code/coreai/ondevice/artifacts/qwen2_5_omni_3b_audio_encoder_fp16_k15.aimodel"
    #else
        private var modelsDir: URL {
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("models")
        }
        private var decoderDir: String {
            modelsDir.appendingPathComponent("qwen2_5_omni_3b_thinker_n750_ios").path
        }
        private var encoderModel: String {
            modelsDir.appendingPathComponent("qwen2_5_omni_3b_audio_encoder_fp16_k15.aimodel").path
        }
    #endif

    @Published var recording = false

    private var model: KitAudioModel?
    private var samples: [Float]?
    private let recorder = MicRecorder()

    func load() async {
        guard !loaded else { return }
        busy = true
        status = "Loading audio model…"
        do {
            let m = try await KitAudioModel(
                decoderBundleAt: URL(fileURLWithPath: decoderDir),
                encoderModelAt: URL(fileURLWithPath: encoderModel),
                arch: .qwen2_5Omni3B)
            self.model = m
            loaded = true
            let mem = availableMemoryMB()
            status = "Model ready (\(mem) MB free). Load audio, then ask."
            log.notice("loaded decoder+encoder; available=\(mem)MB")
        } catch {
            status = "Load failed: \(error.localizedDescription)"
            log.error("load failed: \(error.localizedDescription)")
        }
        busy = false
    }

    func loadFile(_ url: URL) {
        guard let pcm = AudioLoader.load16kMono(url) else {
            status = "Could not decode \(url.lastPathComponent)."
            return
        }
        samples = pcm
        clipName = "\(url.lastPathComponent)  (\(String(format: "%.1f", Double(pcm.count) / 16000))s)"
        answer = ""
        status = "Audio loaded. Ask a question."
    }

    func loadDemoNoise() {
        let pcm = AudioLoader.demoNoise()
        samples = pcm
        clipName = "Demo: white noise (4s)"
        answer = ""
        status = "Demo clip loaded. Ask a question."
    }

    /// Toggle mic recording. On stop, the captured 16 kHz clip becomes the audio to ask about.
    func toggleRecord() {
        if recording {
            recording = false
            status = "Processing recording…"
            recorder.stop { [weak self] pcm in
                Task { @MainActor in
                    guard let self else { return }
                    self.samples = pcm
                    self.clipName = String(format: "Mic clip (%.1fs)", Double(pcm.count) / 16000)
                    self.status = pcm.isEmpty ? "No audio captured." : "Recorded. Ask a question."
                }
            }
        } else {
            recording = true
            answer = ""
            clipName = "Recording… tap Stop when done."
            status = "Listening…"
            recorder.start { [weak self] error in
                Task { @MainActor in
                    guard let self, let error else { return }
                    self.recording = false
                    self.clipName = "No audio loaded."
                    self.status = "Mic error: \(error.localizedDescription)"
                }
            }
        }
    }

    /// Clip cap (s). The decoder prefills the audio tokens one step at a time on device, so a long
    /// clip = a long, slow prefill (the proven demo is 4 s ≈ 100 tokens). Cap for snappy responses.
    private let maxClipSeconds = 6

    func ask(_ question: String) async {
        guard let model, let samples, !busy else { return }
        busy = true
        answer = ""
        let clip = Array(samples.prefix(16000 * maxClipSeconds))
        status = "Encoding audio…"
        do {
            try await model.attach(samples: clip)  // mel → encoder → static buffer
            log.notice("attached audio; available=\(availableMemoryMB())MB")
            status = "Thinking… (on-device, a few seconds)"
            let session = LanguageModelSession(model: model)
            let stream = session.streamResponse(
                to: question.isEmpty ? "What do you hear?" : question,
                options: GenerationOptions(maximumResponseTokens: 128))
            for try await snapshot in stream { answer = snapshot.content }
            status = "Done (\(availableMemoryMB()) MB free)."
            log.notice("answer=\(self.answer, privacy: .public) available=\(availableMemoryMB())MB")
        } catch {
            status = "Generation failed: \(error.localizedDescription)"
            log.error("generation failed: \(error.localizedDescription)")
        }
        busy = false
    }

    /// Headless device measurement: load → demo noise → ask, logging memory at each phase to the
    /// unified log (os.Logger "com.coreaikit.audiochat/selftest"). Launch the app with `--selftest`.
    func selfTest() async {
        log.notice("SELFTEST start; available=\(availableMemoryMB())MB")
        await load()
        guard loaded else { log.error("SELFTEST abort: model not loaded"); return }
        loadDemoNoise()
        await ask("What do you hear?")
        log.notice("SELFTEST done; answer=\(self.answer, privacy: .public)")
    }
}
