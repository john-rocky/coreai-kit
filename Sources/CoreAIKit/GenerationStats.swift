import Darwin
import Foundation

/// Live performance numbers for the current/last generation.
public struct GenerationStats: Sendable, Equatable {
    public var loadSeconds: Double? = nil
    public var promptTokens: Int = 0
    /// Leading prompt tokens served from the engine's KV cache this turn (the engine's
    /// prefix hit count — 0 on engines that reuse internally without reporting).
    public var cachedPromptTokens: Int = 0
    public var ttftSeconds: Double? = nil
    public var generatedTokens: Int = 0
    /// Decode speed over a 32-token rolling window. The engine bursts at decode start, so a
    /// cumulative average over-reads on short replies; the window converges to steady state.
    public var tokensPerSecond: Double? = nil
    public var footprintBytes: UInt64 = 0

    public init() {}
}

enum ProcessStats {
    static func seconds(
        from start: SuspendingClock.Instant, to end: SuspendingClock.Instant
    ) -> Double {
        let d = end - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }
}
