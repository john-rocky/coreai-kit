// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the Forecast runner: `swift run forecast-cli --series series.csv` forecasts on
// macOS with no Xcode and no device — the agent-verifiable door. It compiles the same Sources/QuickStart.swift
// and Sources/SeriesInput.swift the GUI app (project.yml / Forecast.xcodeproj) ships.
let package = Package(
    name: "Forecast",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "forecast-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            sources: ["Sources/QuickStart.swift", "Sources/SeriesInput.swift", "CLI/main.swift"]
        )
    ]
)
