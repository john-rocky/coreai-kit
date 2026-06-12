// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DocChat",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "DocChat",
            dependencies: [
                .product(name: "CoreAIKit", package: "coreai-kit"),
                .product(name: "CoreAIKitEmbeddings", package: "coreai-kit"),
            ]
        )
    ]
)
