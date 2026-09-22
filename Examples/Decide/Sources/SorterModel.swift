// SorterModel — screen (e): a folder sorted by meaning. Every file is read as text (plain,
// Markdown, PDF, or an image through Vision), prefilled once, and asked two questions: which
// of the named bins it belongs in (choice) and what it needs from you (choice whose first
// option is "nothing"). Apply moves the files into one subfolder per bin. No index, no
// rules, no generation — one scored prompt per question, and every file's milliseconds on
// screen.
//
// The second question is a choice, not a yes/no, on purpose: asked "does this need
// something from you?" a chat model says yes to a manual (it does tell you to do things);
// asked to pick between "nothing: a receipt, a manual, terms to keep" and the kinds of ask,
// the same model separates them (12/12 on the sample folder, MiniCPM5 2B).

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class SorterModel {
    struct Bin: Identifiable, Hashable {
        let name: String
        let description: String
        var id: String { name }
    }

    struct Entry: Identifiable {
        let id = UUID()
        let url: URL
        let excerpt: String
        var bin: Decision.Answer?
        var need: Decision.Answer?
        /// The id of the "nothing" option, so `needsYou` knows which answer means no ask.
        var nothing: String?

        var name: String { url.lastPathComponent }
        var chosen: String? { bin?.choice }
        var needLabel: String? { need?.choice }
        var needsYou: Bool { need != nil && need?.choice != nothing }
        var milliseconds: Double { (bin?.timing.milliseconds ?? 0) + (need?.timing.milliseconds ?? 0) }
    }

    /// Characters read per file: the head of a document says what it is.
    static let excerptLimit = 1500

    var folder: URL?
    var binsText = SorterModel.sampleBins
    /// `question | nothing option | ask | ask…` — the first option is what "no ask" reads as.
    var needText = "What does this document need from the reader? | nothing: it is a receipt, a manual, terms to keep, or a note | a payment | a reply, signature or decision by a date"
    var entries: [Entry] = []
    var status = "Open a folder, or make the sample folder, then Sort."
    var working = false
    var applied = false
    private var scopedFolder: URL?

    var totalMilliseconds: Double { entries.map(\.milliseconds).reduce(0, +) }
    var sorted: Int { entries.filter { $0.bin != nil }.count }
    var flagged: Int { entries.filter(\.needsYou).count }

    /// The need question from `needText`; nil unless it has a question and two options.
    var needQuestion: (question: Decision.Question, nothing: String)? {
        let parts = needText.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count >= 3, !parts[0].isEmpty else { return nil }
        let options = Array(parts.dropFirst()).filter { !$0.isEmpty }
        guard options.count >= 2 else { return nil }
        return (.choice(parts[0], options), options[0])
    }

    /// One bin per line: `name: what goes in it`. The model reads both; the answer reports the name.
    static func parseBins(_ text: String) -> [Bin] {
        text.components(separatedBy: "\n").compactMap { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { return nil }
            if let colon = trimmed.firstIndex(of: ":") {
                let name = trimmed[..<colon].trimmingCharacters(in: .whitespaces)
                let description = trimmed[trimmed.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                return Bin(name: name, description: description.isEmpty ? name : description)
            }
            return Bin(name: trimmed, description: trimmed)
        }
    }

    var bins: [Bin] { Self.parseBins(binsText) }

    func open(_ url: URL) {
        if let scopedFolder { scopedFolder.stopAccessingSecurityScopedResource() }
        scopedFolder = url.startAccessingSecurityScopedResource() ? url : nil
        folder = url
        applied = false
        list()
    }

    /// Writes the sample files into a fresh folder under the temporary directory and opens it.
    func makeSampleFolder() {
        let dir = FileManager.default.temporaryDirectory.appending(path: "Decide sorter sample", directoryHint: .isDirectory)
        do {
            try? FileManager.default.removeItem(at: dir)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for (name, text) in Self.sampleFiles {
                try text.write(to: dir.appending(path: name), atomically: true, encoding: .utf8)
            }
            binsText = Self.sampleBins
            open(dir)
        } catch {
            status = "Could not write the sample folder: \(error.localizedDescription)"
        }
    }

    private func list() {
        guard let folder else { return }
        entries = []
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])) ?? []
        for url in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            guard let text = DocumentText.read(url, limit: Self.excerptLimit) else { continue }
            entries.append(Entry(url: url, excerpt: text))
        }
        status = entries.isEmpty
            ? "No readable files in \(folder.lastPathComponent)."
            : "\(entries.count) readable files in \(folder.lastPathComponent) — Sort."
    }

    func sort(_ runtime: DecideRuntime) {
        guard !working, !entries.isEmpty else { return }
        let bins = bins
        guard bins.count >= 2 else {
            status = "Name at least two bins, one per line."
            return
        }
        guard let need = needQuestion else {
            status = "The need line wants a question and at least two options, separated by |."
            return
        }
        working = true
        applied = false
        for index in entries.indices {
            entries[index].bin = nil
            entries[index].need = nil
            entries[index].nothing = need.nothing
        }
        let binQuestion = Decision.Question.choice(
            "Which folder does this document belong in?",
            options: bins.map { .init(id: $0.name, description: "\($0.name): \($0.description)") })
        Task {
            defer { working = false }
            do {
                let decider = try await runtime.ready()
                for index in entries.indices {
                    status = "Deciding \(entries[index].name)…"
                    let prefilled = try await decider.prefill(entries[index].excerpt)
                    entries[index].bin = try await prefilled.decide(binQuestion)
                    entries[index].need = try await prefilled.decide(need.question)
                }
                status = "\(entries.count) files, 2 decisions each, in \(ms(totalMilliseconds)) · \(flagged) need something from you · Apply moves them"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }

    /// Moves every sorted file into `<folder>/<bin>/`.
    func apply() {
        guard let folder, !working, sorted > 0 else { return }
        var moved = 0
        do {
            for index in entries.indices {
                guard let bin = entries[index].chosen else { continue }
                let target = folder.appending(path: bin, directoryHint: .isDirectory)
                try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
                let destination = target.appending(path: entries[index].name)
                try FileManager.default.moveItem(at: entries[index].url, to: destination)
                entries[index] = Entry(
                    url: destination, excerpt: entries[index].excerpt,
                    bin: entries[index].bin, need: entries[index].need, nothing: entries[index].nothing)
                moved += 1
            }
            applied = true
            status = "Moved \(moved) files into \(Set(entries.compactMap(\.chosen)).count) folders under \(folder.lastPathComponent) · \(entries.count * 2) decisions in \(ms(totalMilliseconds))"
        } catch {
            status = "Move failed after \(moved) files: \(error.localizedDescription)"
        }
    }

    static let sampleBins = """
        Receipts: invoices, receipts, payment confirmations and reminders
        Contracts: agreements, terms and conditions, leases, warranties
        Manuals: setup guides, instructions, troubleshooting steps
        Letters: personal letters, cards, notes between people
        """

    /// Twelve invented documents: three per bin, some that ask for something and some that do not.
    static let sampleFiles: [(String, String)] = [
        ("invoice-0417.txt", """
            INVOICE #0417
            From: Harbor Lane Roasters
            To: Unit 4B, 12 Maple Court
            2 kg espresso blend .......... 84.00
            Delivery ..................... 6.00
            Amount due: 90.00
            Due date: 5 October. Please pay by bank transfer, quoting the invoice number. \
            A late fee applies after the due date.
            """),
        ("receipt-grinder.txt", """
            Thank you for your purchase.
            Item: Burr coffee grinder, black, 1 unit
            Paid: 129.00 by card ending 4421
            Order 88213 · Delivered 14 September
            This receipt is for your records. No further action is needed.
            """),
        ("payment-reminder.md", """
            # Second reminder — account 2291

            Our records show a balance of 48.50 from the August statement is still unpaid.
            Please settle the balance within 7 days to avoid a service interruption.
            If you have already paid, reply to this notice with the payment date.
            """),
        ("lease-renewal.md", """
            # Lease renewal — 12 Maple Court, Unit 4B

            Cedar Ridge Properties offers to renew the lease for a further twelve months at 1,480 per month, \
            all other terms unchanged. To accept, sign both copies and return one to the office by 30 September. \
            If we do not hear from you by then, the current lease ends on its stated date.
            """),
        ("nda-mutual.md", """
            # Mutual non-disclosure agreement

            Each party agrees to keep the other party's confidential information secret, to use it only for \
            evaluating the proposed collaboration, and to return or destroy it on request. This agreement \
            lasts three years from the date of signature. It does not create any obligation to do business.
            """),
        ("warranty-terms.txt", """
            LIMITED WARRANTY
            This appliance is warranted against defects in materials and workmanship for 24 months from the \
            date of purchase. The warranty does not cover damage from misuse, liquids, or unauthorised repair. \
            Keep your receipt; it is required for any claim.
            """),
        ("router-quick-start.txt", """
            QUICK START — home router
            1. Connect the WAN port to your modem with the supplied cable.
            2. Plug in the power adapter and wait for the status light to turn solid white.
            3. On your phone, join the network printed on the label and open the setup page.
            4. Choose a new network name and password, then save.
            """),
        ("grinder-cleaning.md", """
            # Cleaning the burr grinder

            Unplug the grinder. Turn the hopper counter-clockwise and lift it off. Remove the upper burr by \
            turning the ring to the unlock mark. Brush both burrs and the chute; do not wash them. \
            Reassemble in reverse order and run a handful of beans through before use.
            """),
        ("printer-troubleshooting.txt", """
            TROUBLESHOOTING — paper jams
            If the printer reports a jam: open the rear cover, pull the sheet out slowly in the paper path \
            direction, and check for torn pieces. Close the cover; the print job resumes. \
            Frequent jams usually mean curled or damp paper — store paper flat in its packaging.
            """),
        ("letter-from-june.txt", """
            Dear neighbour,

            Thank you again for watering the plants while we were away — everything survived, even the basil. \
            We're having a small get-together on the 12th around six and would love you to come. \
            Let us know by Friday if you can make it so we know how many chairs to borrow.

            Warmly, June
            """),
        ("thank-you-card.txt", """
            Just a note to say thank you for the lovely dinner last week. The dessert recipe you mentioned \
            turned out perfectly. Hope to see you again soon. — Tom & Ana
            """),
        ("note-to-self.md", """
            Ideas from the walk: repaint the hallway a lighter colour, move the bookshelf next to the window, \
            and finally frame the two prints from the trip. Maybe a Sunday project.
            """),
    ]
}
