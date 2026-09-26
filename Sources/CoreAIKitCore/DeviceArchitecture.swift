// DeviceArchitecture.swift — which iPhone generation this is, in Core AI's name for it, for the
// one decision the store makes with it: whether a repo's `ios-<arch>/` is this device's
// (`ModelID.subtrees`). The only use of Core AI in this target, which runs no model: it reads a
// name, on iOS alone. The default store is created in this target, so the name cannot be handed
// in from a target above.

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
