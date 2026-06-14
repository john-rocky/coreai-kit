// swift-tools-version: 6.0
import PackageDescription

// The runnable, testable core of the SiriAsk example: deterministic retrieval + grounding +
// the model host, plus a headless gate harness and the security unit tests. The Siri / App
// Intents / SwiftUI layer lives in `App/` (an XcodeGen project that also compiles these core
// sources) — App Intents and onscreen awareness need an app bundle, but the model story and the
// 347 mitigations are provable here with `swift run` / `swift test`, no device required.
let package = Package(
    name: "SiriAsk",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .target(
            name: "SiriAskCore",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")]
        ),
        .executableTarget(
            name: "SiriAskGate",
            dependencies: [
                "SiriAskCore",
                .product(name: "CoreAIKit", package: "coreai-kit"),
            ]
        ),
        .testTarget(
            name: "SiriAskCoreTests",
            dependencies: ["SiriAskCore"]
        ),
    ]
)
