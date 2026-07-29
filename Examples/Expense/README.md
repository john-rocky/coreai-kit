# Expense — a receipt photo becomes a Swift value

Point it at a receipt, get a typed struct back. No server, no API key, no per-scan cost, and the
photo never leaves the phone — so there is nothing to declare about where the data went.

The whole thing is two calls:

```swift
let text = try await CoreAI.read(image)                      // the page, structure intact
let receipt = try await CoreAI.extract(text, as: Receipt.self)  // your type, filled in
```

`Receipt` is an ordinary struct you write:

```swift
@Generable
struct Receipt {
    @Guide(description: "Merchant or store name, as printed") var merchant: String
    @Guide(description: "Grand total as a number, no currency symbol") var total: Double
    @Guide(description: "Date in YYYY-MM-DD") var date: String
    @Guide(description: "One-word expense category") var category: String
}
```

`@Generable` is Apple's own macro. The framework derives a schema from it and constrains
generation to that schema, so you get this type or an error — never a string to parse and hope
about.

## Run

```bash
swift run expense-cli --image receipt.jpg          # terminal, macOS
swift run expense-cli --image receipt.jpg --json

xcodegen generate                                  # app
open Expense.xcodeproj
```

`Sources/QuickStart.swift` is the part to copy. Both the CLI and the app call it and nothing
else.

## What it costs

**About 4.1 GB on first use** — GLM-OCR (1.6 GB) reads the page, and a chat model (2.5 GB) fills
the struct. That is a product decision, not an afternoon: budget for the download, fetch it
behind a first-run screen with `CoreAI.prepare(.read, .extract)`, and expect it once per install.

Worth knowing what the 1.6 GB buys, because Apple's `VNRecognizeTextRequest` is free and already
on the device: Vision returns the *words*. This returns the *structure* — a receipt's line items
stay in their table instead of collapsing into a stream of text, which is what makes the total
findable rather than guessable. If your documents are plain prose, Vision is the better trade.

## Notes

- Runs offline. Airplane mode is a good way to prove to yourself where the photo went.
- The category is generated, not chosen from a list. If you need a fixed taxonomy, say so in the
  `@Guide` description and validate it after — the model is constrained to the *shape*, not to
  your business rules.
- Accuracy on crumpled thermal paper is a real question and this example does not answer it.
  Test on your own receipts before promising anything to a finance team.
