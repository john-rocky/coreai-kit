// swift-tools-version: 6.0
import PackageDescription

// Compile-check door: `swift build` builds the full SwiftUI app as a macOS executable
// with no Xcode and no device. The GUI app ships via project.yml / OpsGallery.xcodeproj
// (xcodegen generate) from the same Sources.
let package = Package(
    name: "OpsGallery",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "opsgallery",
            dependencies: [.product(name: "CoreAIOps", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "project.yml", "README.md", "OpsGallery.entitlements"],
            sources: ["Sources"]
        )
    ]
)
