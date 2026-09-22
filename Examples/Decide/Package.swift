// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the Decide runner: `swift run decide-cli ask --state "…" --noul "…"`
// scores typed questions on macOS with no Xcode and no device — the agent-verifiable door.
// It compiles the same Sources/QuickStart.swift the GUI app (project.yml / Decide.xcodeproj)
// ships, plus `bench` (shared vs direct prefill) and `oracle` (agreement with a reference
// row file) for the numbers in the README.
let package = Package(
    name: "Decide",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(name: "coreai-kit", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "decide-cli",
            dependencies: [.product(name: "CoreAIOps", package: "coreai-kit")],
            path: ".",
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
