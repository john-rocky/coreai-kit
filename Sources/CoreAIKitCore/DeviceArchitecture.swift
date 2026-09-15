// DeviceArchitecture.swift — which iPhone generation this is, in Core AI's name for it, for the
// one decision the store makes with it: whether a repo's `ios-<arch>/` is this device's
// (`ModelID.subtrees`). The only use of Core AI in this target, which runs no model: it reads a
// name, on iOS alone. The default store is created in this target, so the name cannot be handed
// in from a target above.
//
// The catalog's device-specific variant keys (`ios-ane-<arch>`, `CatalogEntry.variantKey`) are
// resolved from a separate table, `DeviceArchitecture.validated`: a key is taken only on device
// identifiers that have loaded that kind of bundle on hardware, never from the naming rule alone.

import Foundation

#if os(iOS) && canImport(CoreAI)
import CoreAI
#endif

extension ModelStore {
    /// The architecture Core AI compiles for on this iPhone (`h18p` on an iPhone 17 Pro, `h19p`
    /// on an 18 Pro), read once, on the first download or cache lookup of a model whose path is
    /// `ios`. nil on the Mac, whose `macos/` graphs it specializes itself, below iOS 27, and where
    /// Core AI is absent.
    static let deviceArchitecture: String? = {
        #if os(iOS) && canImport(CoreAI)
        if #available(iOS 27, *) { return AIModel.deviceArchitectureName }
        #endif
        return nil
    }()
}

public enum DeviceArchitecture {
    /// The `coreai-build` architecture id of the running device for the catalog's device-specific
    /// variant keys, or nil when it is not known for certain: a Mac, the Simulator, or an iPhone
    /// the table below has not been validated on. nil selects the portable ("ios") variant.
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
