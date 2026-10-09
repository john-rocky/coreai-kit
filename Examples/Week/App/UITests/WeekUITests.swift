// WeekUITests — accepts the two system alerts the app raises when it asks (full Calendar access,
// then Reminders), so a phone nobody touches can run the demo afterwards. It launches the app with
// `-grantOnly 1` (the app asks for both, shows the answer and writes Documents/access.json), taps
// every alert button whose label starts with "Allow" ("Allow Full Access"), and passes when the
// app reports full access to both. The test process cannot read the app's Documents on a phone, so
// it reads the same answer from the app's screen.
//
//   xcodebuild test -project Examples/Week/App/Week.xcodeproj -scheme Week \
//       -destination "id=<device udid>" -only-testing:WeekUITests/WeekUITests/testGrantAccess

import XCTest

final class WeekUITests: XCTestCase {
    @MainActor
    func testGrantAccess() throws {
        let allow = NSPredicate(format: "label BEGINSWITH 'Allow'")
        let monitor = addUIInterruptionMonitor(withDescription: "privacy alert") { alert in
            let button = alert.buttons.matching(allow).firstMatch
            guard button.exists else { return false }
            button.tap()
            return true
        }
        defer { removeUIInterruptionMonitor(monitor) }

        let app = XCUIApplication()
        app.launchArguments = ["-grantOnly", "1", "-log", "1"]
        app.launch()
        let status = app.staticTexts["access-status"]
        XCTAssertTrue(status.waitForExistence(timeout: 30), "the app did not show its access line")

        #if os(iOS)
        // The alerts belong to SpringBoard; the monitor above only sees one on the app's next
        // interaction, so both ways are tried until the app reports the answer.
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        #endif
        let deadline = Date().addingTimeInterval(120)
        while Date() < deadline, !Self.granted(status.label) {
            #if os(iOS)
            let button = springboard.alerts.buttons.matching(allow).firstMatch
            if button.waitForExistence(timeout: 2) {
                button.tap()
                continue
            }
            app.tap()
            #else
            Thread.sleep(forTimeInterval: 1)
            #endif
        }
        XCTAssertTrue(Self.granted(status.label), "the app reports \(status.label)")
    }

    static func granted(_ label: String) -> Bool {
        label.contains("events=fullAccess") && label.contains("reminders=fullAccess")
    }
}
