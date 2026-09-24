// DecisionCalibrationTests.swift — the catalog's calibration record and what the kit does with
// it, checkable without weights: the record decodes (its provenance ignored), the temperature
// per question type resolves in its order, a distribution re-read at another temperature is the
// same logits' softmax, and the fitting arithmetic is Examples/Decide/conformance/calibration.py's.

import Foundation
import Testing

@testable import CoreAIKit

struct CatalogCalibrationTests {
    @Test func aRecordDecodesItsTemperaturesAndIgnoresItsProvenance() throws {
        let json = """
            {"version": 1, "models": [
              {"id": "one", "name": "One", "repo": "org/one", "kind": "chat",
               "variants": {"macos": {"path": "int8"}},
               "calibration": {"temperature": 2.34,
                 "fit": {"fixture": "SemIf perturbations108", "rows": 108, "questionTypes": ["choice"]},
                 "report": {"fixture": "SemIf authored144", "rows": 144, "eceBefore": 0.167, "eceAfter": 0.078}}},
              {"id": "typed", "name": "Typed", "repo": "org/typed", "kind": "chat",
               "variants": {"macos": {"path": "int8"}},
               "calibration": {"temperature": 2, "byType": {"noul": 1.5, "score": 3}}},
              {"id": "none", "name": "None", "repo": "org/none", "kind": "decision",
               "variants": {"macos": {"path": "int8"}}}
            ]}
            """
        let catalog = try JSONDecoder().decode(ModelCatalog.self, from: Data(json.utf8))
        let one = try #require(catalog.entry(id: "one")?.calibration)
        #expect(one == CatalogEntry.Calibration(temperature: 2.34))
        #expect(one.byType == nil && one.temperature(forType: "noul") == 2.34)
        let typed = try #require(catalog.entry(id: "typed")?.calibration)
        #expect(typed.temperature == 2 && typed.byType == ["noul": 1.5, "score": 3])
        #expect(typed.temperature(forType: "choice") == 2)
        #expect(typed.temperature(forType: "noul") == 1.5 && typed.temperature(forType: "score") == 3)
        #expect(catalog.entry(id: "none")?.calibration == nil)
        let reencoded = try JSONDecoder().decode(CatalogEntry.Calibration.self, from: JSONEncoder().encode(typed))
        #expect(reencoded == typed)
    }

    @Test func aRecordWithoutItsOneTemperatureDoesNotDecode() {
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(CatalogEntry.Calibration.self, from: Data(#"{"byType": {"choice": 2}}"#.utf8))
        }
    }

    /// A record sits only on a model without a temperature of its own: an author's — a decider
    /// card, a bundle's declaration, one folded into the weights — is kept.
    @Test func shippedRecordsArePositiveAndOnlyOnModelsWithoutTheirOwnTemperature() {
        for entry in ModelCatalog.builtin.models {
            guard let calibration = entry.calibration else { continue }
            let values = [calibration.temperature] + (calibration.byType.map { Array($0.values) } ?? [])
            #expect(values.allSatisfy { $0.isFinite && $0 > 0 }, "\(entry.id)")
            #expect(Set((calibration.byType ?? [:]).keys).isSubset(of: ["choice", "score", "noul"]), "\(entry.id)")
            let format = entry.format ?? (entry.kind == .decision ? "decider" : "chat")
            #expect(!["decider", "slot", "scalar", "letterList", "decisionFunction"].contains(format), "\(entry.id)")
        }
    }
}

struct TemperatureOrderTests {
    static let slot = SlotPrompt.Layout(slots: 256, abstainSlot: 255, choice: 1.055, score: 1.008, noul: 1.047)

    static func all(_ t: Double) -> DecisionTemperatures {
        DecisionTemperatures(choice: t, score: t, noul: t)
    }

    @available(macOS 27, iOS 27, *)
    func resolve(
        _ configured: Double? = nil, _ catalog: CatalogEntry.Calibration? = nil, format: Decision.Format,
        slot: SlotPrompt.Layout? = nil, scalar: ScalarPrompt.Layout? = nil, letters: LetterListPrompt.Layout? = nil
    ) throws -> DecisionTemperatures {
        try TypedDecisions.resolveTemperatures(
            id: "m", configured: configured, catalog: catalog, format: format, slot: slot, scalar: scalar,
            letters: letters)
    }

    @available(macOS 27, iOS 27, *)
    @Test func theConfiguredTemperatureWinsForEveryType() throws {
        #expect(try resolve(0.5, .init(temperature: 2, byType: ["noul": 1.5]), format: .slot, slot: Self.slot) == Self.all(0.5))
        #expect(try resolve(1, .init(temperature: 2.34), format: .chat) == Self.all(1))
    }

    @available(macOS 27, iOS 27, *)
    @Test func thenTheCatalogItsTypeBeforeItsOneValue() throws {
        let t = try resolve(nil, .init(temperature: 2, byType: ["noul": 1.5]), format: .slot, slot: Self.slot)
        #expect(t == DecisionTemperatures(choice: 2, score: 2, noul: 1.5))
        #expect(t.temperature(for: .noul(yes: nil, no: nil)) == 1.5)
        #expect(t.temperature(for: .score(levels: ["low", "high"])) == 2)
        #expect(t.temperature(for: .choice([.init("a"), .init("b")])) == 2)
        #expect(try resolve(nil, .init(temperature: 2.34), format: .chat) == Self.all(2.34))
        #expect(try resolve(nil, .init(temperature: 0.9), format: .decider) == Self.all(0.9))
    }

    @available(macOS 27, iOS 27, *)
    @Test func thenTheBundleDeclarationThenTheFormDefault() throws {
        #expect(try resolve(format: .slot, slot: Self.slot) == DecisionTemperatures(choice: 1.055, score: 1.008, noul: 1.047))
        #expect(try resolve(format: .scalar, scalar: ScalarPrompt.Layout(temperature: 1.75, maxLength: 384)) == Self.all(1.75))
        #expect(try resolve(format: .letterList, letters: LetterListPrompt.Layout(temperature: 0.85)) == Self.all(0.85))
        #expect(try resolve(format: .decider) == Self.all(1.03))
        for format in [Decision.Format.chat, .sharedState, .decisionFunction] {
            #expect(try resolve(format: format) == Self.all(1))
        }
        // A declaration counts only for the form that reads it.
        #expect(try resolve(format: .chat, slot: Self.slot) == Self.all(1))
    }

    @available(macOS 27, iOS 27, *)
    @Test func aCatalogTemperatureThatIsNotPositiveFailsTheLoad() throws {
        for bad in [0, -1, Double.nan, .infinity] {
            #expect(throws: DecisionError.self) { try resolve(nil, .init(temperature: bad), format: .chat) }
        }
        #expect(
            throws: DecisionError.unsupportedModel(
                id: "m", reason: "its catalog entry declares a calibration temperature that is not a positive number")
        ) { try resolve(nil, .init(temperature: 2, byType: ["noul": 0]), format: .chat) }
        // A type this kit never asks is not its to judge.
        #expect(try resolve(nil, .init(temperature: 2, byType: ["rank": -1]), format: .chat) == Self.all(2))
    }
}

struct RescaleTests {
    @Test func aDistributionReReadAtAnotherTemperatureIsTheSameLogitsSoftmax() {
        let logits: [Double] = [2.5, -1.25, 0.3, 4.1, -7.0]
        for from in [1, 1.03, 2.34] {
            let read = DecisionPrompt.probabilities(logits: logits, temperature: from)
            for to in [1, 0.5, 1.75, 2.34, 11.9] {
                let expected = DecisionPrompt.probabilities(logits: logits, temperature: to)
                let got = DecisionCalibration.rescale(read, from: from, to: to)
                #expect(zip(got, expected).allSatisfy { abs($0 - $1) < 1e-12 }, "\(from) → \(to)")
                #expect(DecisionCalibration.argmax(got) == 3)
            }
        }
        #expect(DecisionCalibration.rescale([], to: 2).isEmpty)
    }

    @Test func calibrationPysScaleAgrees() {
        // calibration.py `scale([0.90, 0.07, 0.03], e)`.
        let p = DecisionCalibration.rescale([0.90, 0.07, 0.03], to: exp(1))
        let expected = [0.5963153535104687, 0.23304755118112588, 0.17063709530840543]
        #expect(zip(p, expected).allSatisfy { abs($0 - $1) < 1e-12 })
    }

    @Test func aZeroProbabilityIsFloored() {
        let p = DecisionCalibration.rescale([1, 0], to: 2)
        #expect(p.allSatisfy(\.isFinite) && abs(p.reduce(0, +) - 1) < 1e-12 && p[1] > 0)
    }
}

struct CalibrationArithmeticTests {
    /// Nine labelled rows with their options shuffled per row, as SemIf's are; one row ties
    /// its first two options. The expected numbers below are calibration.py's on the same rows
    /// (`fit_temperature`, `metrics`, its table), to six places.
    static let rows: [DecisionCalibration.Row] = {
        let (s, i, c) = ("supported", "insufficient", "contradicted")
        let table: [(String, [String], [Double], Int)] = [
            ("a", [s, i, c], [0.90, 0.07, 0.03], 0),
            ("a", [i, c, s], [0.85, 0.10, 0.05], 1),
            ("a", [c, s, i], [0.60, 0.30, 0.10], 0),
            ("a", [s, c, i], [0.20, 0.75, 0.05], 0),
            ("b", [i, s, c], [0.70, 0.20, 0.10], 0),
            ("b", [c, i, s], [0.95, 0.03, 0.02], 2),
            ("b", [s, i, c], [0.40, 0.35, 0.25], 1),
            ("b", [c, s, i], [0.10, 0.80, 0.10], 1),
            ("b", [i, c, s], [0.45, 0.45, 0.10], 1),
        ]
        return table.map { DecisionCalibration.Row(probabilities: $0.2, label: $0.3, optionIDs: $0.1, family: $0.0) }
    }()

    static func check(
        _ m: DecisionCalibration.Metrics, n: Int, accuracy: Double, balanced: Double, nll: Double, brier: Double,
        ece: Double, confidence: Double, sourceLocation: SourceLocation = #_sourceLocation
    ) {
        func close(_ a: Double, _ b: Double) -> Bool { abs(a - b) < 1e-6 }
        #expect(m.n == n, sourceLocation: sourceLocation)
        #expect(close(m.accuracy, accuracy), "accuracy \(m.accuracy)", sourceLocation: sourceLocation)
        #expect(close(m.balancedAccuracy, balanced), "balanced \(m.balancedAccuracy)", sourceLocation: sourceLocation)
        #expect(close(m.nll, nll), "nll \(m.nll)", sourceLocation: sourceLocation)
        #expect(close(m.brier, brier), "brier \(m.brier)", sourceLocation: sourceLocation)
        #expect(close(m.ece, ece), "ece \(m.ece)", sourceLocation: sourceLocation)
        #expect(close(m.meanConfidence, confidence), "confidence \(m.meanConfidence)", sourceLocation: sourceLocation)
    }

    @Test func theFittedTemperatureIsCalibrationPys() {
        let grid = DecisionCalibration.temperatureGrid
        #expect(grid.count == 160 && grid.first == exp(-60.0 / 40) && grid.last == exp(99.0 / 40))
        #expect(DecisionCalibration.fitTemperature(Self.rows) == exp(40.0 / 40))
        #expect(DecisionCalibration.fitTemperature([]) == 1)
    }

    @Test func theMetricsAsReportedAreCalibrationPys() {
        let table = DecisionCalibration.familyMetrics(Self.rows)
        #expect(table.map(\.family) == ["a", "b", "all"])
        Self.check(table[0].metrics, n: 4, accuracy: 0.5, balanced: 0.5, nll: 1.132052, brier: 0.75395, ece: 0.525, confidence: 0.775)
        Self.check(table[1].metrics, n: 5, accuracy: 0.4, balanced: 0.333333, nll: 1.268034, brier: 0.64476, ece: 0.46, confidence: 0.66)
        // `all` pools the rows but averages the families' balanced accuracy, as the script does.
        Self.check(table[2].metrics, n: 9, accuracy: 0.444444, balanced: 0.416667, nll: 1.207598, brier: 0.693289, ece: 0.355556, confidence: 0.711111)
        #expect(abs(DecisionCalibration.metrics(Self.rows).balancedAccuracy - 0.444444) < 1e-6)
    }

    @Test func theMetricsAtTheFittedTemperatureAreCalibrationPys() {
        let t = DecisionCalibration.fitTemperature(Self.rows)
        let scaled = Self.rows.map { row in
            var row = row
            row.probabilities = DecisionCalibration.rescale(row.probabilities, to: t)
            return row
        }
        let table = DecisionCalibration.familyMetrics(scaled)
        Self.check(table[0].metrics, n: 4, accuracy: 0.5, balanced: 0.5, nll: 0.97433, brier: 0.599555, ece: 0.304305, confidence: 0.522435)
        Self.check(table[1].metrics, n: 5, accuracy: 0.4, balanced: 0.333333, nll: 1.054336, brier: 0.634139, ece: 0.482709, confidence: 0.478613)
        Self.check(table[2].metrics, n: 9, accuracy: 0.444444, balanced: 0.416667, nll: 1.018778, brier: 0.618768, ece: 0.296296, confidence: 0.498089)
    }

    @Test func balancedAccuracyCountsTheRightOptionsIdNotItsPosition() {
        // The same rows without ids fall back to the position, which the shuffle scrambles.
        let positional = Self.rows.map { row in
            var row = row
            row.optionIDs = nil
            return row
        }
        #expect(abs(DecisionCalibration.metrics(positional).balancedAccuracy - 0.333333) < 1e-6)
        // A tie goes to the first option, as Python's `max` over the indices does.
        #expect(DecisionCalibration.argmax([0.45, 0.45, 0.10]) == 0)
        #expect(DecisionCalibration.metrics([]).n == 0 && DecisionCalibration.metrics([]).nll.isNaN)
    }
}
