// AudioChatApp.swift — entry point for the on-device audio-understanding demo.

import SwiftUI

@main
struct AudioChatApp: App {
    var body: some Scene {
        WindowGroup("AudioChat") {
            AudioChatView()
        }
        .defaultSize(width: 560, height: 520)
    }
}
