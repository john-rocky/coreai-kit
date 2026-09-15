import XCTest

@testable import CoreAIKitCore

/// The architecture table decides which devices download a device-specific (AOT) catalog
/// variant. A wrong row costs a multi-gigabyte download followed by `invalidCompiledModel`, so
/// only identifiers that have loaded an AOT bundle on hardware may map; everything else is nil.
final class DeviceArchitectureTests: XCTestCase {
    func testOnlyValidatedIdentifierMajorsMap() {
        XCTAssertEqual(DeviceArchitecture.architecture(forMachine: "iPhone18,1"), "h18p")
        XCTAssertEqual(DeviceArchitecture.architecture(forMachine: "iPhone18,3"), "h18p")
        // No validated row: portable variant.
        XCTAssertNil(DeviceArchitecture.architecture(forMachine: "iPhone17,1"))
        XCTAssertNil(DeviceArchitecture.architecture(forMachine: "iPad16,3"))
        // A Mac and the Simulator report the CPU, not a device identifier.
        XCTAssertNil(DeviceArchitecture.architecture(forMachine: "arm64"))
        XCTAssertNil(DeviceArchitecture.architecture(forMachine: ""))
    }

    func testMacReportsNoArchitecture() {
        #if os(macOS)
        XCTAssertNil(DeviceArchitecture.current)
        #endif
    }
}
