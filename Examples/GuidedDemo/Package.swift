// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GuidedDemo",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "GuidedDemo",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")]
        )
    ]
)
