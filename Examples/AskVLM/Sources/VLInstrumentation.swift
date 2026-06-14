// VLInstrumentation.swift — memory + latency instrumentation for the on-device VLM running inside
// the system Visual Intelligence query (a background launch with an undocumented memory budget).
// `phys_footprint` is what jetsam accounts; `os_proc_available_memory()` (iOS) is the headroom left
// before this process is killed. Logged to the unified log, which survives a jetsam.

import Foundation
import os

#if canImport(Darwin)
import Darwin
#endif

enum MemoryProbe {
    static let log = Logger(subsystem: "com.coreaikit.askvlm", category: "vlm-vi")

    /// `phys_footprint` in MB — the value jetsam accounts per process.
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

    /// MB remaining before this process hits its per-process jetsam limit (iOS); 0 elsewhere.
    static func availableMB() -> Double {
        #if os(iOS)
        return Double(os_proc_available_memory()) / 1_048_576
        #else
        return 0
        #endif
    }

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

    static func snapshotLine() -> String {
        let f = footprintMB()
        let a = availableMB()
        return a > 0
            ? String(format: "%.0f MB used · %.0f MB headroom", f, a)
            : String(format: "%.0f MB used", f)
    }
}
