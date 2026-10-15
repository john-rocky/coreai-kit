// LookUITests — the screen driven by its own controls, the way a person uses it: wait for READY,
// tap the sign example, tap the first example question, tap Ask, wait for DONE, and keep the numbers
// line (and the answer as the screen shows it) as attachments of the test.
//
//   xcodebuild test -project Look.xcodeproj -scheme Look -destination 'platform=iOS,id=<udid>'
//
// iPhone only (the Take a photo button and the recording are iPhone things). The app is started with
// `-log 1`, so it also writes its own record of the answer (Documents/look-run-<epoch>.json, the same
// keys as autoplay's). The first run of a UI test on a phone asks for its passcode on the phone.

import XCTest

final class LookUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testSignQuestion() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-log", "1"]
        app.launch()

        // The model loads as the app starts: READY (the first launch after an install also downloads
        // and specializes it, so the wait is long).
        let pill = app.staticTexts["state.pill"]
        XCTAssertTrue(pill.waitForExistence(timeout: 30), "the state pill is not on the screen")
        wait(for: pill, label: ["READY"], timeout: 240)

        app.buttons["example.sign"].tap()
        app.buttons["chip.0"].tap()
        let ask = app.buttons["ask"]
        XCTAssertTrue(ask.isEnabled, "Ask is not enabled after the photo and the question")
        ask.tap()

        wait(for: pill, label: ["DONE", "STOPPED", "FAILED"], timeout: 90)
        XCTAssertEqual(pill.label, "DONE", "the answer did not finish")

        let stats = app.descendants(matching: .any)["stats.line"]
        XCTAssertTrue(stats.exists, "the numbers line is not on the screen")
        let answer = app.descendants(matching: .any)["answer.text"]
        let numbersLine = stats.label
        let answerShown = answer.exists ? answer.label : ""
        XCTContext.runActivity(named: "The numbers line and the answer") { activity in
            let line = XCTAttachment(string: numbersLine)
            line.name = "stats.line"
            line.lifetime = .keepAlways
            activity.add(line)
            let shown = XCTAttachment(string: answerShown)
            shown.name = "answer.text"
            shown.lifetime = .keepAlways
            activity.add(shown)
        }
        XCTAssertTrue(numbersLine.hasPrefix("first token "), "numbers line: \(numbersLine)")
    }

    /// Waits until the element's label is one of `label`.
    @MainActor
    private func wait(for element: XCUIElement, label: [String], timeout: TimeInterval) {
        let predicate = NSPredicate(format: "label IN %@", label as NSArray)
        let expectation = XCTNSPredicateExpectation(predicate: predicate, object: element)
        let result = XCTWaiter().wait(for: [expectation], timeout: timeout)
        XCTAssertEqual(result, .completed, "waited \(Int(timeout)) s for \(label), the state is \(element.label)")
    }
}
