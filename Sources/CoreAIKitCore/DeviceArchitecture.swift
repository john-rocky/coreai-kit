// DeviceArchitecture.swift — which AOT architecture id this device can load.
//
// `xcrun coreai-build compile --architecture <id>` emits one `.aimodelc` per id (two ids =
// two bundles, never one bundle for both), and a device loads only the id that matches it:
// an `h17p` bundle on an iPhone 17 Pro fails with `invalidCompiledModel`. The id follows the
// device identifier's major version (`iPhone18,1` → `h18p`), not the marketing name, and the
// CoreAI framework offers no query for it. So the catalog resolves device-specific variants
// from `hw.machine` against a table of identifiers that have loaded an AOT bundle on hardware,
// and answers nil — portable variants only — for everything else. nil is never a guess.

import Foundation

public enum DeviceArchitecture {
    /// The `coreai-build` architecture id of the running device, or nil when it is not known
    /// for certain: a Mac, the Simulator, or an iPhone the table below has not been validated
    /// on. nil selects the portable ("ios") variant.
    public static let current: String? = {
        #if os(iOS)
        return architecture(forMachine: machine())
        #else
        return nil
        #endif
    }()

    /// Device identifier major → architecture id. One row per generation that has loaded an
    /// AOT bundle on hardware; a row is not added from the naming rule alone.
    ///
    /// - `iPhone18`: iPhone 17 Pro (`iPhone18,1`) loads and runs `h18p` bundles (zoo
    ///   `knowledge/aot-and-specialization.md`, device-validated 2026-06-10; the MiniCPM5
    ///   Neural Engine gate ran on it on the iOS 27 RC, 2026-09-15). The other `iPhone18,x`
    ///   identifiers are mapped by the same major-version rule and have not been loaded
    ///   individually — `ChatSession(catalog:)` falls back to the portable variant if one
    ///   refuses the bundle.
    static let validated: [String: String] = [
        "iPhone18": "h18p",
    ]

    /// The devices that load an architecture's bundles, in words a user reads. Follows
    /// `validated` (the `iPhone18` row covers the iPhone 17 family by the same rule).
    static let deviceNames: [String: String] = [
        "h18p": "iPhone 17 models",
    ]

    /// nil when this device loads `.aimodelc` bundles compiled for `architecture`; otherwise
    /// one sentence naming the devices that do. Callers ask before downloading, so a model whose
    /// iOS files are AOT-only for another chip is refused up front instead of failing after the
    /// download with Core AI's `AIModelError error 0`.
    public static func unsupportedReason(forAOT architecture: String, model name: String) -> String? {
        if current == architecture { return nil }
        let devices = deviceNames[architecture] ?? "other devices"
        return "\(name)'s iPhone version is built for \(devices) only."
    }

    /// Maps a `hw.machine` string ("iPhone18,1") to an architecture id, or nil.
    static func architecture(forMachine machine: String) -> String? {
        guard let comma = machine.firstIndex(of: ",") else { return nil }
        return validated[String(machine[..<comma])]
    }

    /// `hw.machine`: the device identifier on iOS ("iPhone18,1"); "arm64" on a Mac or in the
    /// Simulator, which the table above does not contain.
    static func machine() -> String {
        var size = 0
        guard sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname("hw.machine", &buffer, &size, nil, 0) == 0 else { return "" }
        return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
    }
}
