// Inbox.swift — a synthetic customer-support inbox for the demo: an online shop that also sells an
// app subscription. Each message is one intent's template with its slots filled, an optional opener
// and closer that carry the tone and the urgency, and an optional first-name sign-off.
//
// Nothing in it names a real company, product or person: products are plain item names ("electric
// kettle"), people appear only as a first name under a message, and no message holds an email
// address, phone number, street address or URL. An order number is the only identifier.
//
// A seed gives the same inbox on every platform and toolchain: every draw is SplitMix64 through
// `below(_:)`, never the standard library's random algorithms.

import Foundation

/// One inbox message.
public struct Message: Sendable, Identifiable, Hashable {
    /// Position in the inbox, from 0 (the screen numbers it from 1: #0001).
    public let id: Int
    public let text: String
    /// What the generator wrote the message as. Only for reading the model's answers against: it
    /// is never shown and never given to the model.
    public let written: Written

    public struct Written: Sendable, Hashable {
        public let intent: String
        public let urgency: String
        public let sentiment: String
    }
}

public enum Inbox {
    /// `count` messages from `seed`. The same seed and count give the same messages; a longer
    /// inbox from the same seed starts with the shorter one.
    public static func generate(count: Int, seed: UInt64) -> [Message] {
        var rng = SplitMix64(seed: seed)
        return (0..<max(0, count)).map { message(id: $0, rng: &rng) }
    }

    /// Words per message (whitespace-separated), the unit the 6...45 bound is kept in.
    public static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
    }

    public static let wordRange = 6...45

    // MARK: - one message

    enum Tone: String, Sendable { case positive, neutral, negative }

    enum Urgency: String, Sendable { case low, normal, high, critical }

    /// Which tones a body can be written in: a report of something broken is not cheerful, and a
    /// how-do-I question is not angry.
    enum Mood: Sendable { case any, upset, calm }

    /// A body for one intent. `tone` / `urgency`: what the body says by itself (a damaged item is
    /// negative, "not at all" is critical); nil leaves them to the message's draw.
    struct Template: Sendable {
        let text: String
        var mood = Mood.any
        var tone: Tone? = nil
        var urgency: Urgency? = nil
    }

    static func message(id: Int, rng: inout SplitMix64) -> Message {
        let intent = intents[rng.below(intents.count)]
        let template = rng.pick(intent.templates)
        let body = fill(template.text, rng: &rng)

        let tone = template.tone ?? intent.tones.filter { t in
            switch template.mood {
            case .any: return true
            case .upset: return t != .positive
            case .calm: return t != .negative
            }
        }.draw(&rng)

        var urgency = template.urgency ?? intent.urgencies.draw(&rng)
        if tone == .negative, urgency == .low { urgency = .normal }  // nobody angry writes "no rush"
        var urgencyPhrase: String?
        if template.urgency == nil, intent.urgencyPhrases {
            switch urgency {
            case .low: urgencyPhrase = rng.pick(lowPhrases)
            case .normal: urgencyPhrase = nil
            case .high: urgencyPhrase = rng.pick(highPhrases)
            case .critical: urgencyPhrase = rng.pick(criticalPhrases)
            }
        }

        // opener: a sentence in the message's tone, else sometimes a greeting
        var opener: String?
        switch tone {
        case .positive:
            opener = rng.chance(0.6) ? rng.pick(intent.fans ? fanOpeners : politeOpeners) : nil
        case .negative: opener = rng.chance(0.6) ? rng.pick(negativeOpeners) : nil
        case .neutral: opener = nil
        }
        let toneOpener = opener != nil
        if opener == nil, rng.chance(0.4) { opener = rng.pick(greetings) }
        // closer in the tone, when the opener did not already carry it (a body with a tone of its
        // own carries it too)
        var closer: String?
        switch tone {
        case .positive: closer = toneOpener || (template.tone == .positive && rng.chance(0.5)) ? nil : rng.pick(positiveClosers)
        case .negative: closer = toneOpener || (template.tone == .negative && rng.chance(0.5)) ? nil : rng.pick(negativeClosers)
        case .neutral: closer = rng.chance(0.25) ? rng.pick(neutralClosers) : nil
        }
        // no "Thank you. Thanks, Ben"
        let signoffPool = closer == nil ? signoffs : signoffs.filter { $0 != "Thanks," }
        var signoff: String? = rng.chance(0.45) ? rng.pick(signoffPool) + " " + rng.pick(names) : nil

        func assemble() -> String {
            var parts: [String] = []
            var first = body
            if let opener {
                parts.append(opener)
                // "Hi, my order …" — a greeting runs into the body
                if opener.hasSuffix(","), !(first.hasPrefix("I ") || first.hasPrefix("I'")) {
                    first = first.prefix(1).lowercased() + first.dropFirst()
                }
            }
            parts.append(first)
            if let urgencyPhrase { parts.append(urgencyPhrase) }
            if let closer { parts.append(closer) }
            if let signoff { parts.append(signoff) }
            return parts.joined(separator: " ")
        }
        var text = assemble()
        // Keep within wordRange: drop the least informative part first. The tone and urgency
        // phrases stay unless nothing else is left to drop.
        for drop in 0..<3 where wordCount(text) > wordRange.upperBound {
            switch drop {
            case 0: signoff = nil
            case 1: if toneOpener { closer = nil } else { opener = nil }
            default: opener = nil
            }
            text = assemble()
        }

        return Message(
            id: id, text: text,
            written: .init(intent: intent.label, urgency: urgency.rawValue, sentiment: tone.rawValue))
    }

    /// Replaces every {slot} with a draw from its vocabulary.
    static func fill(_ template: String, rng: inout SplitMix64) -> String {
        var out = ""
        var rest = Substring(template)
        while let open = rest.firstIndex(of: "{"), let close = rest[open...].firstIndex(of: "}") {
            out += rest[..<open]
            let slot = rest[rest.index(after: open)..<close]
            switch slot {
            case "order": out += String(10_000 + rng.below(90_000))
            case "amount": out += "$\(9 + rng.below(241))\(rng.pick([".99", ".49", ".00", ".95"]))"
            case "product": out += rng.pick(products)
            case "gadget": out += rng.pick(gadgets)
            case "device": out += rng.pick(devices)
            case "assembled": out += rng.pick(assembled)
            case "sized": out += rng.pick(sized)
            case "small": out += rng.pick(small)
            case "plan": out += rng.pick(plans)
            case "date": out += rng.pick(dates)
            case "period": out += rng.pick(periods)
            case "years": out += rng.pick(years)
            case "deadline": out += rng.pick(deadlines)
            case "screen": out += rng.pick(screens)
            case "editScreen": out += rng.pick(editScreens)
            case "wish": out += rng.pick(wishes)
            case "wishVerb": out += rng.pick(wishVerbs)
            default: out += "{\(slot)}"
            }
            rest = rest[rest.index(after: close)...]
        }
        return out + rest
    }

    // MARK: - intents

    struct Weighted<T: Sendable & Equatable>: Sendable {
        let items: [(T, Int)]

        func draw(_ rng: inout SplitMix64) -> T {
            var k = rng.below(items.reduce(0) { $0 + $1.1 })
            for (item, w) in items {
                if k < w { return item }
                k -= w
            }
            return items[items.count - 1].0
        }

        func filter(_ keep: (T) -> Bool) -> Weighted {
            let kept = items.filter { keep($0.0) }
            return kept.isEmpty ? self : Weighted(items: kept)
        }
    }

    struct Intent: Sendable {
        let label: String
        let templates: [Template]
        var tones = Weighted<Tone>(items: [(.positive, 25), (.neutral, 45), (.negative, 30)])
        var urgencies = Weighted<Urgency>(items: [(.low, 12), (.normal, 60), (.high, 16), (.critical, 12)])
        /// Positive openers from `fanOpeners` ("Big fan of the app!") instead of `politeOpeners`
        /// ("Thank you in advance!").
        var fans = false
        /// Whether "No rush." / "ASAP" phrases are added for the drawn urgency.
        var urgencyPhrases = true
    }

    static let intents: [Intent] = [
        Intent(label: "order_status", templates: [
            .init(text: "My order #{order} still says processing. It was placed on {date}. When will it ship?"),
            .init(text: "I ordered the {product} on {date} and the tracking page has not updated in {period}. Where is it?", mood: .upset),
            .init(text: "Could you check on order #{order}? It was due on {date} and nothing has arrived yet.", mood: .upset),
            .init(text: "Has order #{order} shipped yet? I need the {product} before {deadline}."),
            .init(text: "Tracking says my {product} was delivered, but there is nothing at my door. Order #{order}.", mood: .upset),
            .init(text: "Just checking on order #{order}. I placed it {period} ago and have not had a shipping email."),
            .init(text: "Any update on when my {product} will be sent out?", mood: .calm),
            .init(text: "I got a confirmation for order #{order} but no tracking number yet. When does it ship?", mood: .calm),
            .init(text: "Where is order #{order}? The {product} was supposed to arrive {period} ago.", mood: .upset),
        ]),
        Intent(label: "refund_request", templates: [
            .init(text: "I returned the {product} {period} ago and still have not received my refund of {amount}. Order #{order}.", mood: .upset),
            .init(text: "The {product} arrived damaged. I would like a full refund, please. Order #{order}.", tone: .negative),
            .init(text: "Please refund the {amount} for order #{order}. The {product} is not what was described.", mood: .upset),
            .init(text: "Can I get my money back for the {gadget}? It stopped working after {period}.", mood: .upset),
            .init(text: "I sent the {product} back on {date}. When will the {amount} reach my card?"),
            .init(text: "The {sized} I received is the wrong size. I want a refund, not an exchange.", mood: .upset),
            .init(text: "I cancelled order #{order} within an hour, but I was still charged {amount}. Please refund it.", mood: .upset),
            .init(text: "My {assembled} came with a missing part. I would rather have a refund than wait for a replacement.", mood: .upset),
        ]),
        Intent(label: "cancel_subscription", templates: [
            .init(text: "Please cancel my {plan}. I no longer need it."),
            .init(text: "How do I cancel my subscription? I do not want to be charged again on {date}."),
            .init(text: "I want to end my {plan} before it renews. Please confirm the cancellation."),
            .init(text: "Cancel my membership, please. I am switching to something else."),
            .init(text: "I tried to cancel my {plan} in the app, but the button does nothing. Please cancel it for me.", mood: .upset),
            .init(text: "Please stop my monthly delivery box after this month."),
            .init(text: "I started the free trial by mistake. Please cancel it before it turns into a paid {plan}.", mood: .calm),
            .init(text: "We are moving abroad, so please close my {plan} at the end of this billing period.", mood: .calm),
        ], tones: Weighted(items: [(.positive, 15), (.neutral, 50), (.negative, 35)])),
        Intent(label: "technical_issue", templates: [
            .init(text: "The app crashes every time I open {screen}.", mood: .upset),
            .init(text: "Since the last update, {screen} shows a blank page on my phone.", mood: .upset),
            .init(text: "My {device} will not pair with the app. I have restarted both several times.", mood: .upset),
            .init(text: "I get an error every time I try to upload a photo. The spinner never stops.", mood: .upset),
            .init(text: "The save button in {editScreen} is greyed out, so I cannot keep my changes.", mood: .upset),
            .init(text: "The app freezes on the loading screen right after I log in.", mood: .upset),
            .init(text: "Sync between my phone and tablet stopped working on {date}.", mood: .upset),
            .init(text: "The firmware update on my {device} failed halfway, and now it will not turn on at all.", mood: .upset, urgency: .critical),
            .init(text: "The app does not open at all since this morning.", mood: .upset, urgency: .critical),
        ]),
        Intent(label: "billing_question", templates: [
            .init(text: "Why was I charged {amount} on {date}? I thought my plan was {amount} a month.", mood: .upset),
            .init(text: "Can you explain the extra {amount} fee on my last invoice?"),
            .init(text: "How do I update the card on my account before the next payment?", mood: .calm),
            .init(text: "Is there a discount if I switch from the monthly plan to the annual plan?", mood: .calm),
            .init(text: "Where can I download an invoice for my {plan} payment?", mood: .calm),
            .init(text: "My statement shows two charges from you this month. What is each one for?"),
            .init(text: "Does the {plan} price include tax, or is that added at checkout?", mood: .calm),
            .init(text: "I paid part of order #{order} with a gift card, but the full {amount} went on my card. Can you check?", mood: .upset),
        ]),
        Intent(label: "account_access", templates: [
            .init(text: "I cannot log in to my account. The password reset email never arrives.", mood: .upset),
            .init(text: "My account is locked after too many login attempts. How do I unlock it?"),
            .init(text: "I lost my phone and cannot get the login code. Please help me get back in."),
            .init(text: "The app says my username does not exist, but I have had an account for {years}.", mood: .upset),
            .init(text: "I changed my password yesterday and now it accepts neither the old one nor the new one.", mood: .upset),
            .init(text: "I got an alert about a sign-in from a device I do not own. Is my account safe?", mood: .calm, urgency: .high),
            .init(text: "The login link in your email says it has expired every time I tap it.", mood: .upset),
            .init(text: "I can't get into my account at all, and my order history is in there.", mood: .upset, urgency: .critical),
        ]),
        Intent(label: "feature_request", templates: [
            .init(text: "It would be great if the app had {wish}."),
            .init(text: "Could you add a way to {wishVerb}? I would use it every day."),
            .init(text: "Please consider adding {wish} in a future update."),
            .init(text: "Any plans to support {wish}? It is the one thing I miss."),
            .init(text: "I would love to be able to {wishVerb} from the home screen."),
            .init(text: "Suggestion: let us {wishVerb}. It would save me a lot of time."),
            .init(text: "Will the {product} ever come in more colors? A dark green one would be perfect."),
            .init(text: "Adding {wish} would make the app so much better."),
        ], tones: Weighted(items: [(.positive, 45), (.neutral, 55)]),
           urgencies: Weighted(items: [(.low, 35), (.normal, 65)]), fans: true),
        // Feedback asks for nothing: written as low urgency, with no urgency phrase.
        Intent(label: "general_feedback", templates: [
            .init(text: "Just wanted to say the {product} is even better than I expected.", tone: .positive),
            .init(text: "The new design of the app is clean and easy to use.", tone: .positive),
            .init(text: "The packaging was far too much for such a small {small}. Maybe use less plastic?", tone: .negative),
            .init(text: "The delivery driver was friendly and careful with the box.", tone: .positive),
            .init(text: "I have used the app for {years} now, and it is part of my daily routine.", tone: .positive),
            .init(text: "The instructions that came with the {assembled} were confusing, but I figured it out.", tone: .neutral),
            .init(text: "Overall a good experience, though checkout felt a bit slow.", tone: .neutral),
            .init(text: "The {product} looks nothing like the photos in your shop. Not impressed.", tone: .negative),
        ], urgencies: Weighted(items: [(.low, 1)]), fans: true, urgencyPhrases: false),
    ]

    // MARK: - phrases

    static let greetings = ["Hi,", "Hello,", "Hi there,", "Hi team,", "Good morning,", "Hey,"]
    static let politeOpeners = [
        "Thanks for the great service so far.", "You have always been helpful, so I hope you can help again.",
        "Thank you in advance!",
    ]
    /// A happy customer's openers, for a suggestion or a compliment rather than a complaint.
    static let fanOpeners = [
        "Love your shop, by the way.", "Big fan of the app!", "I really enjoy using the app.",
        "Thanks for the great service so far.",
    ]
    static let negativeOpeners = [
        "I am really frustrated.", "This is the third time I am writing about this.",
        "Honestly, this is unacceptable.", "I am very unhappy with how this has gone.", "Really disappointed right now.",
    ]
    static let positiveClosers = ["Thanks a lot!", "Appreciate your help!", "Keep up the good work!", "Thank you so much!"]
    static let negativeClosers = ["Very disappointed.", "Not happy about this.", "Please do better.", "I expected more from you."]
    static let neutralClosers = ["Thanks.", "Thank you."]
    static let lowPhrases = ["No rush.", "Whenever you get a chance.", "Not urgent.", "It can wait."]
    static let highPhrases = [
        "Please reply today.", "I need this sorted soon.", "Please look into this quickly.", "I would appreciate a fast answer.",
    ]
    static let criticalPhrases = [
        "I need this fixed ASAP.", "This is urgent, please help today.", "Please treat this as urgent.",
        "I need an answer right now.",
    ]
    static let signoffs = ["—", "Best,", "Regards,", "Thanks,"]

    // MARK: - vocabulary (made up for this shop; plain item names, no brands)

    static let products = [
        "electric kettle", "bamboo desk lamp", "wool throw blanket", "steel water bottle", "canvas tote bag",
        "wireless earbuds", "smart plug", "standing desk", "coffee grinder", "hiking backpack",
        "linen duvet cover", "cast iron skillet", "yoga mat", "air purifier", "robot vacuum", "phone case",
        "running shoes", "office chair", "baby monitor", "e-reader cover", "fitness tracker", "ring light",
        "kitchen scale", "rain jacket", "electric toothbrush",
    ]
    /// Products that can stop working.
    static let gadgets = [
        "electric kettle", "bamboo desk lamp", "wireless earbuds", "smart plug", "coffee grinder", "air purifier",
        "robot vacuum", "baby monitor", "fitness tracker", "ring light", "kitchen scale", "electric toothbrush",
    ]
    /// Products that pair with the app.
    static let devices = [
        "wireless earbuds", "smart plug", "robot vacuum", "baby monitor", "fitness tracker", "air purifier",
        "kitchen scale", "electric toothbrush",
    ]
    /// Products that come in parts, with instructions.
    static let assembled = [
        "standing desk", "office chair", "coffee grinder", "robot vacuum", "air purifier", "ring light",
        "baby monitor", "bamboo desk lamp",
    ]
    /// Products that come in sizes.
    static let sized = [
        "running shoes", "rain jacket", "linen duvet cover", "phone case", "yoga mat", "hiking backpack", "e-reader cover",
    ]
    /// Products small enough to be over-packed.
    static let small = [
        "phone case", "e-reader cover", "smart plug", "wireless earbuds", "fitness tracker", "electric toothbrush",
    ]
    static let plans = [
        "premium plan", "family plan", "annual plan", "monthly plan", "student plan", "basic plan", "delivery pass",
    ]
    /// Days a thing happened "on".
    static let dates = [
        "March 3", "April 18", "May 14", "June 2", "July 21", "August 9", "September 12", "October 5",
        "November 20", "Monday", "Tuesday", "Sunday", "the 3rd", "the 14th", "the 28th",
    ]
    static let periods = [
        "two days", "three days", "four days", "a week", "ten days", "two weeks", "three weeks", "a month",
    ]
    static let years = ["two years", "three years", "five years", "six months", "over a year"]
    static let deadlines = [
        "Friday", "Saturday", "the weekend", "next Tuesday", "the holidays", "the end of the month",
    ]
    static let screens = [
        "the search screen", "checkout", "my order history", "the wishlist", "the barcode scanner",
        "notifications", "the settings page", "the chat window", "the photo gallery", "saved payment methods",
    ]
    static let editScreens = ["the settings page", "my profile", "the wishlist", "my shopping list", "notification settings"]
    static let wishes = [
        "a dark mode", "a shared shopping list", "an offline mode", "price drop alerts", "a home screen widget",
        "gift wrapping at checkout", "a bigger text size option", "voice search", "reorder reminders",
        "delivery time slots", "a family account", "split payments",
    ]
    static let wishVerbs = [
        "pause my subscription for a month", "reorder a past order in one tap", "save items for later",
        "filter by delivery date", "schedule deliveries for the weekend", "share my wishlist with family",
        "hide items I already bought", "set a monthly budget", "track all my orders on one screen",
    ]
    /// First names only, never with a surname.
    static let names = [
        "Ava", "Ben", "Chloe", "Dev", "Elena", "Felix", "Grace", "Hugo", "Iris", "Jonah", "Kira", "Leo",
        "Mina", "Noah", "Olive", "Priya", "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wren", "Yuki",
        "Zara", "Omar", "Lena", "Marco", "Nadia", "Ravi", "Sofia", "Emil", "Ines", "Tomas", "Aiko", "Bruno",
    ]
}

/// SplitMix64 (Steele, Lea, Flood 2014): a 64-bit state, one add and a mix per draw.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// 0..<n: the high word of next() × n (bias below n / 2^64).
    mutating func below(_ n: Int) -> Int {
        precondition(n > 0)
        return Int(next().multipliedFullWidth(by: UInt64(n)).high)
    }

    mutating func pick<T>(_ xs: [T]) -> T { xs[below(xs.count)] }

    mutating func chance(_ p: Double) -> Bool { Double(next() >> 11) / Double(1 << 53) < p }
}
