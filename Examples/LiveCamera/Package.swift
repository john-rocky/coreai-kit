// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the LiveCamera runner. The camera tabs need a device — the CoreAI
// framework is not in the Simulator SDK and a Mac has no phone camera — but the video scan
// is the same pipeline fed from a file, so it runs here with no Xcode and no device:
//
//   swift run live-cli --video clip.mov --fps 2
//   swift run live-cli --video clip.mov --for person
//   swift run live-cli --bench                       # the pipeline's own per-frame cost
//
// That makes the offline half of this example agent-verifiable, which is the point of every
// CLI shell in Examples/.
let package = Package(
    name: "LiveCamera",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "live-cli",
            dependencies: [.product(name: "CoreAIOps", package: "coreai-kit")],
            path: ".",
            exclude: [
                "LiveCamera.xcodeproj", "project.yml", "README.md",
                "LiveCamera.entitlements",
                // The GUI screens import SwiftUI/UIKit; the CLI compiles QuickStart only.
                "Sources/LiveCameraApp.swift", "Sources/DeviceGate.swift",
                "Sources/CameraPreview.swift", "Sources/DetectScreen.swift",
                "Sources/DepthScreen.swift", "Sources/TriggerScreen.swift",
                "Sources/ScanScreen.swift",
            ],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
