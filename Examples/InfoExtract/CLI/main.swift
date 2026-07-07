// infoextract-cli — argument shell over `InformationExtractor.extract(from:entities:)`: extract PII
// (or any zero-shot label set) from text on-device. Progress goes to stderr so stdout stays
// machine-checkable (agents: assert on stdout).
//
//   swift run infoextract-cli --bundle ~/gliner2_out --text "email me at a@b.com" --labels email
//   swift run infoextract-cli --bundle ~/gliner2_out --gate      # reproduce the demo PII suite
//
// --bundle is a directory holding the `.aimodel` + `tokenizer/` + `extractor.json`.

import CoreAIKitEmbeddings
import Foundation

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func fail(_ s: String) -> Never { err(s); exit(1) }

let usage = """
    usage: infoextract-cli [--bundle <dir>] --text <str> --labels a,b,c [--threshold 0.5]
           infoextract-cli [--bundle <dir>] --gate
    (omit --bundle to download .gliner2PII from the Hugging Face Hub)
    """

var bundle: String?
var text: String?
var labelsCSV = "person,email,phone number,credit card number,social security number,address,date of birth,organization"
var threshold: Float?
var gate = false
var redactMode = false

var args = CommandLine.arguments.dropFirst()
while let a = args.popFirst() {
    switch a {
    case "--bundle": bundle = args.popFirst()
    case "--text": text = args.popFirst()
    case "--labels": labelsCSV = args.popFirst() ?? labelsCSV
    case "--threshold": threshold = args.popFirst().flatMap { Float($0) }
    case "--gate": gate = true
    case "--redact": redactMode = true
    case "-h", "--help": print(usage); exit(0)
    default: fail("unknown arg \(a)\n\(usage)")
    }
}
let labels = labelsCSV.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }

func jsonLine(_ dict: [String: [String]]) -> String {
    let data = try! JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
    return String(data: data, encoding: .utf8)!
}

let extractor: InformationExtractor
if let bundlePath = bundle {
    err("[infoextract] loading bundle \(bundlePath) …")
    extractor = try await InformationExtractor(bundleAt: URL(fileURLWithPath: bundlePath))
} else {
    err("[infoextract] downloading .gliner2PII from the Hub (first run) …")
    extractor = try await InformationExtractor(model: .gliner2PII)
}
err("[infoextract] ready.")

if gate {
    // The GLiNER2-PII demo suite (== ext.extract / export GATE-2/3 ground truth).
    let suite: [(String, [String: [String]])] = [
        ("Contact Dr. Sarah Johnson at sarah.j@acme.com or +1-415-555-0142.",
         ["person": ["Sarah Johnson"], "email": ["sarah.j@acme.com"],
          "phone number": ["+1-415-555-0142"]]),
        ("Wire to account holder James Lee, SSN 123-45-6789, card 4111 1111 1111 1111.",
         ["person": ["James Lee"], "credit card number": ["4111 1111 1111 1111"],
          "social security number": ["123-45-6789"]]),
        ("Send the invoice to 500 Market St, San Francisco and cc maria@corp.io.",
         ["email": ["maria@corp.io"], "address": ["500 Market St, San Francisco"]]),
        ("Patient DOB 1985-07-14, treated at Mercy General Hospital by nurse Alan Poe.",
         ["person": ["Alan Poe"], "date of birth": ["1985-07-14"],
          "organization": ["Mercy General Hospital"]]),
    ]
    var allOK = true
    for (t, expected) in suite {
        let got = try await extractor.extract(from: t, entities: labels)
        let ok = got == expected
        allOK = allOK && ok
        err("\(ok ? "PASS" : "FAIL") | \(t.prefix(46))")
        print(jsonLine(got))
        if !ok { err("  expected: \(jsonLine(expected))") }
    }
    err("\n=== KIT E2E GATE: \(allOK ? "ALL PASS ✅" : "FAILURES ❌") ===")
    exit(allOK ? 0 : 1)
}

guard let t = text else { fail(usage) }
if redactMode {
    let redacted = try await extractor.redact(t, entities: labels, threshold: threshold)
    let spans = try await extractor.extractSpans(from: t, entities: labels, threshold: threshold)
    print(redacted)
    for s in spans.sorted(by: { $0.confidence > $1.confidence }) {
        err(String(format: "  %-22@ %.3f  %@", s.label as NSString, s.confidence, s.text))
    }
} else {
    let got = try await extractor.extract(from: t, entities: labels, threshold: threshold)
    print(jsonLine(got))
}
