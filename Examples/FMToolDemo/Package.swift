// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FMToolDemo",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "FMToolDemo",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")]
        )
    ]
)
