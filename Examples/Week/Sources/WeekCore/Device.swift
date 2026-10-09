// Device.swift — where a run happened, for the footer and the result files: the model identifier,
// the OS, the thermal state, and which bundle answered.

import Foundation

public enum Device {
    /// `utsname.machine`: the model identifier on an iPhone, the CPU architecture on a Mac.
    public static let machine: String = {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: &u.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }()

    /// The model identifier (iPhone19,2 / Mac16,10).
    public static let model: String = {
        #if os(macOS)
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var buffer = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &buffer, &size, nil, 0)
        return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
        #else
        return machine
        #endif
    }()

    /// "iOS 27.0" / "macOS 27.0".
    public static let os: String = {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        #if os(iOS)
        let name = "iOS"
        #else
        let name = "macOS"
        #endif
        let version = v.patchVersion > 0 ? "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)" : "\(v.majorVersion).\(v.minorVersion)"
        return "\(name) \(version)"
    }()

    public static func name(_ state: ProcessInfo.ThermalState?) -> String {
        switch state {
        case .nominal?: return "nominal"
        case .fair?: return "fair"
        case .serious?: return "serious"
        case .critical?: return "critical"
        default: return "unknown"
        }
    }

    /// When a bundle was exported (`compilation.date` in its metadata.json): tells two copies of a
    /// bundle apart when there is no revision to read.
    public static func compiled(_ bundle: URL) -> String? {
        struct Metadata: Decodable {
            struct Compilation: Decodable { let date: String? }
            let compilation: Compilation?
        }
        guard let data = try? Data(contentsOf: bundle.appending(path: "metadata.json")) else { return nil }
        return (try? JSONDecoder().decode(Metadata.self, from: data))?.compilation?.date
    }
}
