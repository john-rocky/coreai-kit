// SearchModel — screen (c): query × passages. One decision per passage — does this passage
// answer the query (noul), or how relevant is it on a three-level scale (score) — and the
// passages ranked by the answer. A reranker made of a chat model: no embedding index, no
// generation, one scored prompt per candidate, and every probability shown with its cost.

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class SearchModel {
    enum Mode: String, CaseIterable {
        case noul = "yes / no"
        case score = "3-level score"
    }

    struct Hit: Identifiable {
        let id = UUID()
        let passage: String
        let answer: Decision.Answer
        /// The ranking key: P(yes), or the expected level over the scale's range.
        var relevance: Double {
            switch answer.value {
            case .noul(let p): return p
            case .score(let s): return s.value / Double(max(1, s.probabilities.count - 1))
            case .choice(let c): return c.confidence
            }
        }
    }

    var query = "How long do I have to return an item for a refund?"
    var passagesText = SearchModel.samplePassages.joined(separator: "\n\n")
    var mode: Mode = .noul
    var hits: [Hit] = []
    var status = "Enter a query and one passage per paragraph, then Rank."
    var working = false

    var totalMilliseconds: Double { hits.map(\.answer.timing.milliseconds).reduce(0, +) }

    static let samplePassages = [
        "Returns are accepted within 30 days of delivery. Items must be unused and in the original packaging; refunds go back to the original payment method within 5 business days of receipt.",
        "Standard shipping takes 3–5 business days. Express shipping is available at checkout for orders placed before 2 pm.",
        "Gift cards are delivered by email within minutes of purchase and never expire. They cannot be exchanged for cash.",
        "To change the delivery address after ordering, contact support before the order ships. Once shipped, the carrier may allow a redirect for a fee.",
        "Warranty claims for manufacturing defects can be made for 12 months from the delivery date. Accidental damage is not covered.",
        "Our stores are open Monday to Saturday, 10 am to 8 pm, and Sundays 11 am to 6 pm. Public holidays may differ.",
    ]

    var passages: [String] {
        passagesText.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    func rank(_ runtime: DecideRuntime) {
        guard !working, !query.isEmpty, !passages.isEmpty else { return }
        working = true
        hits = []
        status = "Ranking…"
        Task {
            defer { working = false }
            do {
                let decider = try await runtime.ready()
                let question: Decision.Question
                switch mode {
                case .noul:
                    question = .noul("Does this passage answer the question: \(query)")
                case .score:
                    question = .score(
                        "How well does this passage answer the question: \(query)",
                        levels: ["not at all", "partly", "fully"])
                }
                var scored: [Hit] = []
                for passage in passages {
                    let answer = try await decider.decide(passage, question)
                    scored.append(Hit(passage: passage, answer: answer))
                    hits = scored.sorted { $0.relevance > $1.relevance }
                }
                status = "\(passages.count) passages in \(ms(totalMilliseconds)) · median \(ms(median(scored.map(\.answer.timing.milliseconds)))) each"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }
}
