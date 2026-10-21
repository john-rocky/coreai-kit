import SwiftUI

@main
struct LookApp: App {
    var body: some Scene {
        WindowGroup {
            // `-watch 1`: the watch mode (Watch.swift), for a check from a script; otherwise the screen.
            if UserDefaults.standard.bool(forKey: "watch") {
                WatchView()
            } else {
                LookView()
            }
        }
    }
}
