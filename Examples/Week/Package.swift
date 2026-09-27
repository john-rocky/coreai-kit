// swift-tools-version: 6.0
import PackageDescription

// `WeekCore` is the week demo's core — the synthetic calendar week and the planner that asks every
// event one typed question on decider-0.8b. `swift run week-cli run` plans a week headless on the
// Mac from the generator or from a file, no EventKit; the SwiftUI app in App/ compiles the same
// sources into itself, so both report numbers from one code path.
let package = Package(
    name: "Week",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .target(
            name: "WeekCore",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")]
        ),
        .executableTarget(
            name: "week-cli",
            dependencies: ["WeekCore", .product(name: "CoreAIKit", package: "coreai-kit")],
            path: "CLI"
        ),
        .testTarget(
            name: "WeekCoreTests",
            dependencies: ["WeekCore"]
        ),
    ]
)
