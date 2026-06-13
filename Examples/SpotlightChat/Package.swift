// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SpotlightChat",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "SpotlightChat",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            linkerSettings: [
                // CoreSpotlight requires the process to have a bundle identity; embedding
                // an Info.plist gives this command-line tool one (no .app wrapper needed).
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "\(Context.packageDirectory)/Sources/SpotlightChat/Info.plist",
                ])
            ]
        )
    ]
)
