// StreamingTagParser.swift — incremental segmenter for a model's text deltas, driven by an
// OutputProfile rule table.
//
// Sections open/close on marker strings that may straddle token boundaries: outside a
// section the parser holds back at most `max(open.count) - 1` trailing characters, inside a
// section at most `max(close.count) - 1`, and emits everything that can be proven not to be
// the start of a marker (the hold-back contract of Apple's ThinkTagParser, generalized).
//
// Feed incremental detokenizer output via `consume(_:)`; call `flush()` once at end of
// stream — an unterminated buffering section (tool call) flushes its partial payload and is
// left to the JSON parser to accept or reject.

struct StreamingTagParser {
    enum Event: Equatable, Sendable {
        case response(String)
        case thinking(String)
        case toolCallPayload(String)
    }

    private let profile: OutputProfile
    private var buffer = ""
    private var activeRule: Int?
    private var pending = ""

    init(profile: OutputProfile) {
        self.profile = profile
    }

    mutating func consume(_ delta: String) -> [Event] {
        buffer.append(delta)
        return drain(isFinal: false)
    }

    mutating func flush() -> [Event] {
        drain(isFinal: true)
    }

    private mutating func drain(isFinal: Bool) -> [Event] {
        var events: [Event] = []
        while true {
            if let index = activeRule {
                let rule = profile.rules[index]

                // Earliest close marker wins.
                var closeHit: Range<String.Index>?
                for marker in rule.close {
                    if let r = buffer.range(of: marker),
                        closeHit == nil || r.lowerBound < closeHit!.lowerBound
                    {
                        closeHit = r
                    }
                }
                if let range = closeHit {
                    emitBody(String(buffer[..<range.lowerBound]), rule, closing: true, &events)
                    buffer = String(buffer[range.upperBound...])
                    activeRule = nil
                    continue
                }

                let safe = isFinal ? buffer.endIndex : lastSafeIndex(forTags: rule.close)
                if safe > buffer.startIndex {
                    emitBody(String(buffer[..<safe]), rule, closing: false, &events)
                    buffer = String(buffer[safe...])
                }
                if isFinal, !pending.isEmpty {
                    events.append(event(rule.kind, pending))
                    pending = ""
                }
                return events
            }

            // Default mode: earliest rule-open wins.
            var hit: (range: Range<String.Index>, rule: Int)?
            for (i, rule) in profile.rules.enumerated() {
                if let r = buffer.range(of: rule.open),
                    hit == nil || r.lowerBound < hit!.range.lowerBound
                {
                    hit = (r, i)
                }
            }
            if let (range, i) = hit {
                let before = String(buffer[..<range.lowerBound])
                if !before.isEmpty, profile.defaultText == .stream {
                    events.append(.response(before))
                }
                buffer = String(buffer[range.upperBound...])
                activeRule = i
                continue
            }

            let safe = isFinal ? buffer.endIndex : lastSafeIndex(forTags: profile.rules.map(\.open))
            if safe > buffer.startIndex {
                let text = String(buffer[..<safe])
                if profile.defaultText == .stream {
                    events.append(.response(text))
                }
                buffer = String(buffer[safe...])
            }
            return events
        }
    }

    private mutating func emitBody(
        _ body: String, _ rule: OutputRule, closing: Bool, _ events: inout [Event]
    ) {
        if rule.streamsBody {
            if !body.isEmpty { events.append(event(rule.kind, body)) }
        } else {
            pending += body
            if closing {
                if !pending.isEmpty { events.append(event(rule.kind, pending)) }
                pending = ""
            }
        }
    }

    private func event(_ kind: OutputRule.Kind, _ body: String) -> Event {
        switch kind {
        case .thinking: return .thinking(body)
        case .response: return .response(body)
        case .toolCall: return .toolCallPayload(body)
        }
    }

    /// Rightmost index such that the suffix from there to end-of-buffer is not a non-empty
    /// prefix of any candidate tag. Scans at most `max(tag.count) - 1` trailing characters.
    private func lastSafeIndex(forTags tags: [String]) -> String.Index {
        let maxHold = (tags.map(\.count).max() ?? 1) - 1
        guard !buffer.isEmpty, maxHold > 0 else { return buffer.endIndex }
        let holdStart = buffer.index(buffer.endIndex, offsetBy: -min(maxHold, buffer.count))
        var idx = holdStart
        while idx < buffer.endIndex {
            let suffix = buffer[idx...]
            if tags.contains(where: { $0.starts(with: suffix) }) {
                return idx
            }
            idx = buffer.index(after: idx)
        }
        return buffer.endIndex
    }
}
