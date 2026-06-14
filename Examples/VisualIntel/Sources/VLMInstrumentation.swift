// VLMInstrumentation.swift — measurement + the experimental flag for running the on-device VLM
// (Qwen3-VL-2B via KitVisionModel) inside the system Visual Intelligence query.
//
// The whole reason this file exists: a Visual Intelligence query runs in a *background launch* of
// the app (App Intents), and Apple publishes NO number for that launch's memory budget. RF-DETR
// nano (~100 MB) survives it on device; a 2 GB+ VLM is the open question. So we MEASURE instead of
// guessing:
//   • `MemoryProbe.mark(_:)` logs `phys_footprint` (the byte count jetsam actually accounts) and,
//     on iOS, `os_proc_available_memory()` (headroom remaining before THIS process is killed) at
//     each stage, to the unified log — which survives a jetsam, so the last breadcrumb before a
//     kill pins where it died. Pair with the system's JetsamEvent report (Settings → Privacy →
//     Analytics Data → JetsamEvent-*.ips) for the authoritative per-process limit at kill time.
//   • `VLMVISettings.runInVisualIntelligence` is a persisted toggle (default OFF) read by the
//     background query, so the VLM path can be A/B-measured on device WITHOUT a rebuild, and the
//     shipped RF-DETR/CLIP demo never regresses when the flag is off.

import Foundation
import os

#if canImport(Darwin)
import Darwin
#endif

/// Lightweight memory instrumentation for the background-launch survival measurement (G1).
enum MemoryProbe {
    static let log = Logger(subsystem: "com.coreaikit.visualintel", category: "vlm-vi")

    /// `phys_footprint` in MB — the value the jetsam mechanism accounts per process (matches what
    /// a JetsamEvent report shows at kill). Works on iOS and macOS.
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    /// Bytes remaining before THIS process hits its per-process jetsam limit, in MB. On iOS this is
    /// exactly the "undocumented background-launch budget" we are trying to read; 0 elsewhere.
    static func availableMB() -> Double {
        #if os(iOS)
        return Double(os_proc_available_memory()) / 1_048_576
        #else
        return 0
        #endif
    }

    /// Log a named stage with footprint (+ iOS headroom). `.public` so it isn't redacted in Console.
    @discardableResult
    static func mark(_ stage: String) -> (footprint: Double, available: Double) {
        let f = footprintMB()
        let a = availableMB()
        if a > 0 {
            log.log(
                "stage=\(stage, privacy: .public) footprint=\(f, format: .fixed(precision: 1))MB available=\(a, format: .fixed(precision: 1))MB")
        } else {
            log.log("stage=\(stage, privacy: .public) footprint=\(f, format: .fixed(precision: 1))MB")
        }
        return (f, a)
    }

    /// A human-readable one-liner of the current memory state, for the in-app mirror (the
    /// foreground control: same VLM, full memory budget).
    static func snapshotLine() -> String {
        let f = footprintMB()
        let a = availableMB()
        return a > 0
            ? String(format: "%.0f MB used · %.0f MB headroom", f, a)
            : String(format: "%.0f MB used", f)
    }
}

/// The experimental "run the VLM inside Visual Intelligence" toggle, persisted so the background VI
/// launch (a separate process launch, same app container) reads the same value the in-app switch
/// wrote. Default OFF = the shipped, proven RF-DETR/CLIP surface is unchanged.
enum VLMVISettings {
    private static let key = "runVLMInVisualIntelligence"

    static var runInVisualIntelligence: Bool {
        get { UserDefaults.standard.bool(forKey: key) }
        set { UserDefaults.standard.set(newValue, forKey: key) }
    }
}
