# ScanToType — a document photo becomes a value of your own type

You write a struct. You point this at a photo. You get the struct back, filled in, on device.

```swift
func scan<T: Generable>(_ image: CGImage, as type: T.Type) async throws -> T {
    let text = try await CoreAI.read(image)          // the page, structure intact
    return try await CoreAI.extract(text, as: type)  // your type, filled in
}
```

Neither call knows what your type is about. **The domain lives entirely in the type you pass**,
so the same two lines are a different app depending on what you hand them:

```swift
@Generable struct Receipt {
    @Guide(description: "Merchant or store name, as printed") var merchant: String
    @Guide(description: "Grand total as a number, no currency symbol") var total: Double
    @Guide(description: "Date in YYYY-MM-DD") var date: String
}

@Generable struct Prescription {
    @Guide(description: "Drug name as printed") var drug: String
    @Guide(description: "Dose with unit, e.g. 5 mg") var dose: String
    @Guide(description: "How often to take it") var frequency: String
}

@Generable struct BusinessCard {
    var name: String
    @Guide(description: "Job title, empty if absent") var title: String
    @Guide(description: "Email address, empty if absent") var email: String
}
```

`try await scan(photo, as: Prescription.self)` and you have a medication scanner. Nothing in
this example changes.

`@Generable` is Apple's own macro. The framework derives a schema from it and constrains
generation to that schema, so the result decodes into your type or fails — there is no string to
parse and hope about.

## Run

```bash
swift run scan-cli --image document.jpg          # terminal, macOS
swift run scan-cli --image document.jpg --json

xcodegen generate                                # app
open ScanToType.xcodeproj
```

`Sources/QuickStart.swift` is the part to copy. The CLI and the app both call it and nothing
else. `Receipt` is in there as one example; replace it with yours.

## What it costs

**About 4.1 GB on first use** — GLM-OCR (1.6 GB) reads the page, and a chat model (2.5 GB) fills
the type. That is a product decision, not an afternoon: fetch it behind a first-run screen with
`CoreAI.prepare(.read, .extract)` and expect it once per install.

Worth knowing what the 1.6 GB buys, because Apple's `VNRecognizeTextRequest` is free and already
on the device: Vision returns the **words**. This returns the **structure** — a table stays a
table instead of collapsing into a stream of text, which is what makes a specific field findable
rather than guessable. **If your documents are plain prose, Vision is the better trade** and you
only need the second call.

## Notes

- Runs offline. Airplane mode is a good way to prove to yourself where the photo went — which
  also means there is no key to store, no per-scan bill, and nothing to declare about where the
  document was processed.
- The model is constrained to the *shape* of your type, not to your business rules. If a field
  must come from a fixed list, say so in its `@Guide` description and validate after.
- Accuracy on a crumpled thermal receipt, a handwritten form, or a bad photo is a real question
  and this example does not answer it. Test on your own documents before promising anything.
