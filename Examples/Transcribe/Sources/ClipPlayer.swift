// ClipPlayer — plays the clip out loud, so whoever watches hears what the model is given. App
// chrome, not model work: nothing here touches the kit.

import AVFoundation

@MainActor
final class ClipPlayer: NSObject, AVAudioPlayerDelegate {
    /// Called when playback ends — played to the end or stopped.
    var onStop: (() -> Void)?

    private var player: AVAudioPlayer?
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Plays `url` from the top and returns its length in seconds.
    func play(_ url: URL) throws -> Double {
        stop()
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)
        #endif
        let player = try AVAudioPlayer(contentsOf: url)
        player.delegate = self
        guard player.play() else { throw ClipPlayerError.couldNotStart(url.lastPathComponent) }
        self.player = player
        return player.duration
    }

    func stop() {
        player?.stop()
        finish()
    }

    /// Returns once the current clip has played to its end or was stopped.
    func finished() async {
        guard player != nil else { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    private func finish() {
        player = nil
        let resumed = waiters
        waiters = []
        for waiter in resumed { waiter.resume() }
        onStop?()
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        let finished = ObjectIdentifier(player)
        Task { @MainActor in
            // Only the clip still playing ends playback; a replaced one is already stopped.
            if let current = self.player, ObjectIdentifier(current) == finished { self.finish() }
        }
    }
}

enum ClipPlayerError: LocalizedError {
    case couldNotStart(String)

    var errorDescription: String? {
        switch self {
        case .couldNotStart(let name): return "Could not play \(name)."
        }
    }
}
