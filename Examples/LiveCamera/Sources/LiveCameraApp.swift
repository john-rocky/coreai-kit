// LiveCameraApp.swift — four tabs, one per live task, so each can be tried on a device
// independently of the others.

import SwiftUI

@main
struct LiveCameraApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                // Launched with `-gate`, the app runs every task once and logs a verdict
                // per task instead of waiting for taps — the device transcript that makes
                // "it works on a phone" checkable. See `DeviceGate`.
                .task {
                    if DeviceGate.isRequested { await DeviceGate.run() }
                }
        }
    }
}

struct ContentView: View {
    var body: some View {
        TabView {
            DetectScreen()
                .tabItem { Label("Detect", systemImage: "viewfinder") }
            DepthScreen()
                .tabItem { Label("Depth", systemImage: "cube.transparent") }
            TriggerScreen()
                .tabItem { Label("Trigger", systemImage: "bell.badge") }
            ScanScreen()
                .tabItem { Label("Scan", systemImage: "film") }
        }
    }
}
