// ChecklistModel — screen (d): one document, many typed questions. The document is prefilled
// once; every question then costs only its own tail (`TypedDecisions.prefill`, then
// `PrefilledState.decide`), and the screen shows exactly that: the prefill's tokens and
// milliseconds, then each decision's milliseconds with the tokens it reused. A checklist over
// a contract, a policy, a report — read once, answered N times, nothing generated.

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class ChecklistModel {
    struct Item: Identifiable {
        let id = UUID()
        let key: String
        let question: Decision.Question
        let answer: Decision.Answer
    }

    /// Characters of a document the checklist reads (about 1,000 tokens — under the iPhone
    /// prompt guard with the questions' tails).
    static let documentLimit = 3600

    var document = ChecklistModel.sampleLease
    var questionsText = ChecklistModel.sampleQuestions
    var items: [Item] = []
    var status = "Paste or open a document, edit the questions, then Run."
    var working = false
    var prefillMilliseconds: Double?
    var prefillTokens = 0
    var documentName = "sample lease"

    var totalMilliseconds: Double { items.map(\.answer.timing.milliseconds).reduce(0, +) }

    /// The level's own words for a score answer, so the list reads "60 days or more", not "level 2".
    func levelName(for score: Decision.Score) -> String? {
        for item in items {
            if case .score(let levels) = item.question.kind, levels.count == score.probabilities.count,
                item.answer.score == score.value { return levels[score.level] }
        }
        return nil
    }
    var medianMilliseconds: Double { median(items.map(\.answer.timing.milliseconds)) }

    /// One question per line: `noul: question`, `choice: question | option | option…`,
    /// `score: question | lowest level | … | highest level`. Blank lines and `#` comments skip.
    static func parse(_ text: String) throws -> [(String, Decision.Question)] {
        var out: [(String, Decision.Question)] = []
        for (index, rawLine) in text.components(separatedBy: "\n").enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            guard let colon = line.firstIndex(of: ":") else {
                throw ParseError.line(index + 1, "expected `noul:`, `choice:` or `score:`")
            }
            let kind = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let parts = line[line.index(after: colon)...].split(separator: "|", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            let question = parts[0]
            let options = Array(parts.dropFirst()).filter { !$0.isEmpty }
            let key = "q\(out.count + 1)"
            switch kind {
            case "noul", "yes/no", "bool":
                out.append((key, .noul(question)))
            case "choice":
                guard options.count >= 2 else { throw ParseError.line(index + 1, "a choice needs at least two options after `|`") }
                out.append((key, .choice(question, options)))
            case "score", "scale":
                guard options.count >= 2 else { throw ParseError.line(index + 1, "a score needs at least two levels after `|`") }
                out.append((key, .score(question, levels: options)))
            default:
                throw ParseError.line(index + 1, "unknown kind `\(kind)`")
            }
        }
        guard !out.isEmpty else { throw ParseError.line(0, "no questions") }
        return out
    }

    enum ParseError: LocalizedError {
        case line(Int, String)
        var errorDescription: String? {
            if case .line(let n, let why) = self { return n == 0 ? why : "line \(n): \(why)" }
            return nil
        }
    }

    func load(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        if let text = DocumentText.read(url, limit: Self.documentLimit) {
            document = text
            documentName = url.lastPathComponent
            items = []
            status = "\(url.lastPathComponent): \(text.count) characters loaded — Run."
        } else {
            status = "No readable text in \(url.lastPathComponent)."
        }
    }

    func loadSample() {
        document = Self.sampleLease
        questionsText = Self.sampleQuestions
        documentName = "sample lease"
        items = []
        status = "Sample loaded — Run."
    }

    func run(_ runtime: DecideRuntime) {
        guard !working, !document.isEmpty else { return }
        let questions: [(String, Decision.Question)]
        do {
            questions = try Self.parse(questionsText)
        } catch {
            status = "Questions: \(error.localizedDescription)"
            return
        }
        working = true
        items = []
        prefillMilliseconds = nil
        status = "Prefilling the document…"
        Task {
            defer { working = false }
            do {
                let decider = try await runtime.ready()
                try await decider.reset()
                let prefilled = try await decider.prefill(document)
                prefillMilliseconds = prefilled.timing.milliseconds
                prefillTokens = prefilled.tokens
                status = "Document prefilled: \(prefilled.tokens) tokens in \(ms(prefilled.timing.milliseconds)) — deciding…"
                for (key, question) in questions {
                    let answer = try await prefilled.decide(question)
                    items.append(Item(key: key, question: question, answer: answer))
                }
                status = "\(items.count) decisions in \(ms(totalMilliseconds)) after a \(ms(prefilled.timing.milliseconds)) prefill of \(prefilled.tokens) tokens · median \(ms(medianMilliseconds)) each"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }

    /// An invented residential lease with a definite answer to every sample question.
    static let sampleLease = """
        RESIDENTIAL LEASE AGREEMENT

        This lease is made between Cedar Ridge Properties (the "Landlord") and the undersigned tenant \
        (the "Tenant") for the apartment at 12 Maple Court, Unit 4B.

        1. Term. The lease runs for twelve (12) months, starting on 1 October and ending on 30 September \
        of the following year. It does not renew automatically; the parties may sign a renewal.

        2. Rent. Rent is 1,450 per month, due on the first day of each month by bank transfer to the \
        account named in Schedule A. Cash and checks are not accepted. Rent is fixed for the whole term \
        and the Landlord may not increase it before the lease ends.

        3. Security deposit. The Tenant has paid a deposit of 1,450. It is refundable within 21 days \
        after the Tenant moves out, less the cost of any damage beyond normal wear and tear.

        4. Utilities. The Tenant pays for electricity, gas, water and internet. The Landlord pays for \
        building insurance and trash collection.

        5. Repairs. The Landlord repairs the heating, plumbing and the appliances supplied with the \
        apartment (refrigerator, stove, dishwasher) at the Landlord's cost, unless the damage was caused \
        by the Tenant's negligence. The Tenant replaces light bulbs and smoke-detector batteries.

        6. Pets. One cat or one dog under 15 kg is allowed with a one-time pet fee of 200. No other \
        animals are permitted.

        7. Subletting. The Tenant may not sublet the apartment or any part of it, or assign this lease, \
        without the Landlord's prior written consent.

        8. Early termination. If the Tenant ends the lease before the term ends, the Tenant owes an \
        early-termination fee equal to two months' rent, unless a replacement tenant approved by the \
        Landlord takes over.

        9. Moving out. The Tenant must give at least sixty (60) days' written notice before moving out \
        at the end of the term.

        10. Insurance. The Tenant must hold renter's insurance with liability cover of at least 100,000 \
        for the whole term and show proof on request.

        11. Smoking. Smoking is not permitted anywhere inside the apartment or in the shared hallways.

        12. Quiet hours. Quiet hours are 10 pm to 7 am.
        """

    static let sampleQuestions = """
        # one question per line — noul: / choice: q | a | b / score: q | low | high
        noul: Is subletting allowed without the landlord's written consent?
        noul: Is there a fee for ending the lease early?
        noul: Are pets allowed?
        noul: Does the tenant pay for electricity?
        noul: Is the security deposit refundable?
        noul: Can the landlord raise the rent during the term?
        noul: Is renter's insurance required?
        choice: What does the lease say about smoking inside the apartment? | it is allowed | it is not allowed | it is not mentioned
        choice: Who pays to repair the dishwasher? | tenant | landlord | shared
        choice: How is rent paid? | bank transfer | cash | check | not stated
        choice: How long is the lease term? | month to month | 6 months | 12 months | 24 months or longer
        score: How much notice must the tenant give before moving out? | none | about 30 days | 60 days or more
        """
}
