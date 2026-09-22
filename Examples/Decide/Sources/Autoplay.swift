// Autoplay — drives one screen hands-off, for a recording or a smoke run from a script:
//
//   Decide.app/Contents/MacOS/Decide -autoplay checklist -model minicpm5-2b -delay 1.5
//
// opens that tab, loads the model, waits `delay` seconds after Ready, then presses the
// screen's own sample button. With `-trigger <path>` it also waits, after Ready, until that
// file exists — a recorder creates it once the capture is rolling. Nothing else changes: the
// screens are the same code with the same buttons; this only presses them.

import Foundation
import Observation

@MainActor
@Observable
final class Autoplay {
    enum Screen: String, CaseIterable {
        case speech, clipboard, search, checklist, sorter
    }

    let screen: Screen?
    let model: String?
    let delay: Double
    let trigger: String?
    private var fired = false

    init() {
        let defaults = UserDefaults.standard  // -key value command-line pairs land here
        screen = defaults.string(forKey: "autoplay").flatMap(Screen.init(rawValue:))
        model = defaults.string(forKey: "model")
        let d = defaults.double(forKey: "delay")
        delay = d > 0 ? d : 1.5
        trigger = defaults.string(forKey: "trigger")
    }

    /// Runs `action` once, on the autoplayed screen only, after the model is ready.
    func run(_ target: Screen, runtime: DecideRuntime, action: @MainActor () async -> Void) async {
        guard screen == target, !fired else { return }
        fired = true
        if let model { runtime.selectedID = model }
        await runtime.load()
        guard runtime.isReady else { return }
        if let trigger {
            while !FileManager.default.fileExists(atPath: trigger) {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        try? await Task.sleep(for: .seconds(delay))
        await action()
    }
}
