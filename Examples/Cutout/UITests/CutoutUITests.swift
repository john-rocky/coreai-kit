// CutoutUITests — the screen through its own controls on an iPhone, and the one step a launch argument cannot take: the
// answer to the system's Photos request. The app runs the lighthouse example with `-autoplay 1 -save 1` (Autoplay.swift);
// when Save asks for Photos access, the test answers the system alert (the first button declines, the last allows; by
// position, so it works in any language) and waits for the app's own sentence for that outcome.
//
//   xcodebuild test -project Cutout.xcodeproj -scheme Cutout -destination 'platform=iOS,id=<udid>' \
//     -only-testing:CutoutUITests/CutoutUITests/testSaveAllowed
//
// The first UI test on a phone asks for the phone's passcode on the phone.

import XCTest

final class CutoutUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    private func launch() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-autoplay", "1", "-example", "lighthouse", "-save", "1", "-log", "1", "-frames", "1", "-gap", "1"]
        app.launch()
        return app
    }

    /// The system's Photos request (the model loads and answers first: up to 600 s on a first launch).
    @MainActor
    private func photosAlert() -> XCUIElement {
        let alert = XCUIApplication(bundleIdentifier: "com.apple.springboard").alerts.firstMatch
        XCTAssertTrue(alert.waitForExistence(timeout: 600), "no Photos request appeared")
        return alert
    }

    /// Waits until the save line holds `text`, then keeps a screenshot of the screen.
    @MainActor
    private func saveLine(_ app: XCUIApplication, contains text: String) {
        let line = app.staticTexts["save.line"]
        let predicate = NSPredicate(format: "label CONTAINS %@", text)
        let result = XCTWaiter().wait(for: [XCTNSPredicateExpectation(predicate: predicate, object: line)], timeout: 60)
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = "save: \(line.exists ? line.label : "-")"
        shot.lifetime = .keepAlways
        add(shot)
        XCTAssertEqual(result, .completed, "the save line is \(line.exists ? line.label : "absent")")
    }

    @MainActor
    func testSaveDenied() throws {
        let app = launch()
        let alert = photosAlert()
        alert.buttons.element(boundBy: 0).tap()
        saveLine(app, contains: "Photos access is off")
    }

    @MainActor
    func testSaveAllowed() throws {
        let app = launch()
        let alert = photosAlert()
        alert.buttons.element(boundBy: alert.buttons.count - 1).tap()
        saveLine(app, contains: "Saved to Photos")
    }
}
