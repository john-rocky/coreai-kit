// UpscaleEngine.swift — downloads AdcSR on first use and runs ×4 super-resolution off the
// main actor via CoreAIKit's SuperResolver.

import CoreAIKitVision
import CoreGraphics
import SwiftUI

@MainActor
final class UpscaleEngine: ObservableObject {
    @Published var status = "Pick a photo to upscale ×4"
    @Published var downloadFraction: Double?
    @Published var busy = false

    private var resolver: SuperResolver?

    private func ensureLoaded() async throws {
        if resolver != nil { return }
        status = "Downloading AdcSR (~1.7 GB)…"
        let r = try await SuperResolver(model: .adcsrX4) { [weak self] p in
            Task { @MainActor in self?.downloadFraction = p.fraction }
        }
        downloadFraction = nil
        resolver = r
    }

    func upscale(_ image: CGImage) async -> CGImage? {
        guard !busy else { return nil }
        busy = true
        defer { busy = false }
        do {
            try await ensureLoaded()
            status = "Upscaling ×4 on-device…"
            let started = Date()
            let out = try await resolver!.upscale(image)
            let secs = Date().timeIntervalSince(started)
            status = String(format: "Done — %d×%d in %.1fs", out.width, out.height, secs)
            return out
        } catch {
            status = "Error: \(error.localizedDescription)"
            return nil
        }
    }
}
