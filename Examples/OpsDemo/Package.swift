// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "OpsDemo",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "OpsDemo",
            dependencies: [.product(name: "CoreAIOps", package: "coreai-kit")]
        )
    ]
)
