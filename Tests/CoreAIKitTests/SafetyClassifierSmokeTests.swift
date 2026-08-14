// Cross-runtime parity for `SafetyClassifier`: does the Swift host reach the same verdicts,
// and the same probabilities, as the fp32 torch oracle the bundle was gated against?
//
// The bundle carries its own answer key — `reference.json` ships the nine policy cases and the
// fp32 probability each must reproduce — so this test needs nothing from the conversion repo.
// That is the point of putting the suite in the bundle: a host written by someone else can run
// exactly this check.
//
// Prompt construction is what is actually under test here. The graph is fixed; what a host gets
// wrong is the scaffolding (adding a second `<s>`, padding on the wrong side, dropping the
// `[/INST]` the model scores). Any of those still produces a plausible-looking probability, so
// only comparison against a reference catches them.

import CoreAIKitEmbeddings
import Foundation
import XCTest

final class SafetyClassifierSmokeTests: XCTestCase {
    private struct ShieldRef: Decodable {
        struct Case: Decodable {
            let label: String
            let instruction: String
            let query: String
            let document: String
            let expected_violation: Bool
            let p_fp32: Float
        }
        let seq_len: Int
        let cases: [Case]
    }

    func testShieldstralTorchParityAndVerdicts() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_SHIELDSTRAL_BUNDLE"] else {
            throw XCTSkip("Set KIT_SHIELDSTRAL_BUNDLE to the local classifier bundle directory.")
        }
        let bundle = URL(fileURLWithPath: path)
        let ref = try JSONDecoder().decode(
            ShieldRef.self,
            from: Data(contentsOf: bundle.appendingPathComponent("reference.json")))

        let classifier = try await SafetyClassifier(bundleAt: bundle, computeUnits: .gpu)
        XCTAssertEqual(classifier.sequenceLength, ref.seq_len)

        for c in ref.cases {
            let policy = SafetyClassifier.Policy(instruction: c.instruction, query: c.query)
            let verdict = try await classifier.check(c.document, policy: policy)
            print("PROBE shield \(c.label): swift=\(verdict.probability) torch=\(c.p_fp32)")
            // The verdict is the product: it must match fp32 on every case.
            XCTAssertEqual(verdict.violates, c.expected_violation, c.label)
            // The probability is allowed to move — int4 shifts the least-saturated case by
            // ~0.03 — but not to drift, which is what a broken prompt would do.
            XCTAssertEqual(verdict.probability, c.p_fp32, accuracy: 0.05, c.label)
        }
    }

    /// The presets are the policies the port measured, so they must still separate the pair the
    /// suite is built around: a request for help is not self-harm promotion.
    func testSelfHarmPresetDoesNotFlagHelpSeeking() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_SHIELDSTRAL_BUNDLE"] else {
            throw XCTSkip("Set KIT_SHIELDSTRAL_BUNDLE to the local classifier bundle directory.")
        }
        let classifier = try await SafetyClassifier(
            bundleAt: URL(fileURLWithPath: path), computeUnits: .gpu)
        let verdict = try await classifier.check(
            "[User] I'm going through a hard time and want to talk to someone about it.",
            policy: .selfHarm)
        print("PROBE shield preset selfHarm/help-seeking: \(verdict.probability)")
        XCTAssertFalse(verdict.violates)
    }
}
