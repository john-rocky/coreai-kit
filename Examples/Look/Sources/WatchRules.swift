// WatchRules — what the watch screen looks for and what it does then. A rule is a condition in
// words ("a chessboard"; the model is asked "Is there a chessboard?"), one reaction, and an SF
// Symbol that shows over the picture whenever the rule fires. The reactions are the app's own
// functions and the system's sounds and voice: a system sound, a haptic, a line spoken by the
// system voice, a timer, a count, a line in the log — or an effect: the picture holds the frame the
// model answered for 3 s and the app edits it at the place the model names, with a short sound
// (WatchEffects.swift; the sound files are fetched, not in the repository, and the chime plays when
// one is missing). An effect shows no symbol and no words.
//
// Rules typed on the screen are kept between launches (UserDefaults). A launch with `-rules`
// (or r5's `-conditions`) uses those instead and keeps nothing:
//
//   -rules "a chessboard=chime|a lighthouse=say:There is the lighthouse|a sign with words on it=count"
//
// One rule per `|`, the condition before the first `=`, then the reaction: chime | glass | bell
// (a sound), haptic, say:<text>, timer:<minutes>, count, log, effect:ripple | effect:zoom |
// effect:spotlight.

import AudioToolbox
import Foundation

/// The sounds a rule can play: three of the system's message tones.
enum WatchSound: String, Codable, CaseIterable, Identifiable {
    case chime, glass, bell

    var id: String { rawValue }

    var title: String { rawValue.capitalized }

    /// The system sound it plays (`AudioServicesPlaySystemSound`).
    var systemSound: SystemSoundID {
        switch self {
        case .chime: 1008
        case .glass: 1009
        case .bell: 1013
        }
    }
}

/// The edits a rule can play on the frame the model answered, where the model says its thing is:
/// the picture holds that frame for 3 s while Core Image edits it (WatchEffects.swift), with a
/// short sound. No image, no symbol, no words.
enum WatchEffect: String, Codable, CaseIterable, Identifiable {
    case ripple, zoom, spotlight

    var id: String { rawValue }

    var title: String { rawValue.capitalized }

    /// Its sound: Sounds/<sound>.m4a in the app (tools/fetch_examples.sh), the chime when the build
    /// does not have it.
    var sound: String {
        switch self {
        case .ripple: "water"
        case .zoom: "horn"
        case .spotlight: "bells"
        }
    }

    /// What happens to the held frame, and its sound, for the editor.
    var look: String {
        switch self {
        case .ripple: "A wave spreads from the place and leaves the picture tinted blue, with the sound of running water."
        case .zoom: "The picture zooms in on the place to twice its size, with a horn."
        case .spotlight: "Everything around the place goes dark and grey, with bells."
        }
    }
}

/// What a rule does when it fires (besides its symbol over the picture, which every rule but an
/// effect shows).
enum WatchReaction: Codable, Equatable {
    case sound(WatchSound)
    case haptic
    case say(String)
    case timer(minutes: Double)
    case count
    case log
    case effect(WatchEffect)

    enum Kind: String, CaseIterable, Identifiable {
        case sound, haptic, say, timer, count, log, effect

        var id: String { rawValue }

        var title: String {
            switch self {
            case .sound: "Sound"
            case .haptic: "Haptic"
            case .say: "Say"
            case .timer: "Timer"
            case .count: "Count"
            case .log: "Log"
            case .effect: "Effect"
            }
        }

        var icon: String {
            switch self {
            case .sound: "speaker.wave.2"
            case .haptic: "iphone.radiowaves.left.and.right"
            case .say: "text.bubble"
            case .timer: "timer"
            case .count: "number"
            case .log: "list.bullet"
            case .effect: "sparkles"
            }
        }
    }

    var kind: Kind {
        switch self {
        case .sound: .sound
        case .haptic: .haptic
        case .say: .say
        case .timer: .timer
        case .count: .count
        case .log: .log
        case .effect: .effect
        }
    }

    /// The record's name for it: "chime", "haptic", "say", "timer", "count", "log", "effect".
    var name: String {
        switch self {
        case .sound(let sound): sound.rawValue
        case .haptic: "haptic"
        case .say: "say"
        case .timer: "timer"
        case .count: "count"
        case .log: "log"
        case .effect: "effect"
        }
    }

    /// The text it speaks, its minutes, or its effect; nil for the others.
    var detail: String? {
        switch self {
        case .say(let text): text
        case .timer(let minutes): Self.minutesText(minutes)
        case .effect(let effect): effect.rawValue
        default: nil
        }
    }

    /// The effect it plays, nil for the other reactions.
    var effect: WatchEffect? {
        if case .effect(let effect) = self { return effect }
        return nil
    }

    /// The `-rules` form: "chime", "say:There is the lighthouse", "timer:3", "effect:ripple".
    var token: String { detail.map { "\(name):\($0)" } ?? name }

    /// Reads one reaction of `-rules`; anything else plays the chime.
    init(token: String) {
        let parts = token.split(separator: ":", maxSplits: 1).map { String($0) }
        let head = (parts.first ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        let rest = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespaces) : ""
        switch head {
        case "haptic", "buzz": self = .haptic
        case "say": self = .say(rest)
        case "timer": self = .timer(minutes: Double(rest).map { min(max($0, 0.1), 120) } ?? 3)
        case "count": self = .count
        case "log": self = .log
        case "sound": self = .sound(WatchSound(rawValue: rest.lowercased()) ?? .chime)
        case "effect": self = .effect(WatchEffect(rawValue: rest.lowercased()) ?? .ripple)
        default:
            if let effect = WatchEffect(rawValue: head) {
                self = .effect(effect)
            } else {
                self = .sound(WatchSound(rawValue: head) ?? .chime)
            }
        }
    }

    /// 3 → "3", 0.5 → "0.5".
    static func minutesText(_ minutes: Double) -> String {
        minutes == minutes.rounded() ? String(Int(minutes)) : String(format: "%g", minutes)
    }

    /// Seconds as a clock: 180 → "3:00", 59.2 → "1:00" (rounded up, so 0:00 only at the end).
    static func clock(_ seconds: Double) -> String {
        let whole = max(0, Int(seconds.rounded(.up)))
        return String(format: "%d:%02d", whole / 60, whole % 60)
    }
}

struct WatchRule: Identifiable, Codable, Equatable {
    var id = UUID()
    var condition: String
    var reaction: WatchReaction
    var symbol: String
}

enum WatchRules {
    /// At most this many rules: one prompt asks about all of them.
    static let maxCount = 6

    /// The symbols the editor offers.
    static let symbols = [
        "checkerboard.rectangle", "light.beacon.max", "signpost.right", "person", "hand.raised", "cup.and.saucer",
        "key", "cat", "dog", "car", "leaf", "eye",
    ]

    /// A new rule's symbol until its editor picks another: an eye, which stands for no thing in
    /// particular.
    static let newSymbol = "eye"

    /// The three rules a first launch starts with: one for each example photo, so Example photos
    /// fires all three.
    static let defaults = [
        WatchRule(condition: "a chessboard", reaction: .sound(.chime), symbol: "checkerboard.rectangle"),
        WatchRule(condition: "a lighthouse", reaction: .say("There is the lighthouse"), symbol: "light.beacon.max"),
        WatchRule(condition: "a sign with words on it", reaction: .count, symbol: "signpost.right"),
    ]

    /// The symbol of a condition the app knows; others take the editor's symbols in turn.
    private static let known = [
        "a chessboard": "checkerboard.rectangle", "a lighthouse": "light.beacon.max",
        "a sign with words on it": "signpost.right", "a person": "person",
    ]

    /// `-rules "cond=reaction|…"` → rules; `named` = `-symbols "a|b|…"`, by place.
    static func parse(_ text: String, symbols named: [String] = []) -> [WatchRule] {
        var rules: [WatchRule] = []
        for item in text.split(separator: "|") {
            let parts = item.split(separator: "=", maxSplits: 1).map { String($0) }
            let condition = (parts.first ?? "").trimmingCharacters(in: .whitespaces)
            guard !condition.isEmpty, rules.count < maxCount else { continue }
            let reaction = parts.count > 1 ? WatchReaction(token: parts[1]) : .sound(.chime)
            rules.append(
                WatchRule(condition: condition, reaction: reaction, symbol: symbol(for: condition, at: rules.count, named: named)))
        }
        return rules
    }

    /// r5's `-conditions "a|b|…"`: each condition plays the chime.
    static func fromConditions(_ conditions: [String], symbols named: [String] = []) -> [WatchRule] {
        conditions.prefix(maxCount).enumerated().map { index, condition in
            WatchRule(condition: condition, reaction: .sound(.chime), symbol: symbol(for: condition, at: index, named: named))
        }
    }

    static func symbol(for condition: String, at index: Int, named: [String] = []) -> String {
        if named.indices.contains(index) { return named[index] }
        return known[condition.lowercased()] ?? symbols[index % symbols.count]
    }

    // MARK: - Kept between launches

    private static let key = "watchRules"

    static func saved() -> [WatchRule]? {
        guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode([WatchRule].self, from: data)
    }

    static func save(_ rules: [WatchRule]) {
        if let data = try? JSONEncoder().encode(rules) { UserDefaults.standard.set(data, forKey: key) }
    }

    // MARK: - What the example photos show

    /// The example photos each known condition is true of (the rest: false).
    private static let shownBy: [String: Set<String>] = [
        "a chessboard": ["object"], "a lighthouse": ["scene"], "a sign with words on it": ["sign"], "a person": [],
    ]

    /// The answers each example photo should get, rule by rule — when every condition is a known
    /// one; nil otherwise.
    static func truth(_ rules: [WatchRule]) -> [String: [String]]? {
        let sets = rules.map { shownBy[$0.condition.lowercased()] }
        guard !rules.isEmpty, sets.allSatisfy({ $0 != nil }) else { return nil }
        var out: [String: [String]] = [:]
        for photo in ["sign", "object", "scene"] {
            out[photo] = sets.map { ($0 ?? []).contains(photo) ? "yes" : "no" }
        }
        return out
    }
}
