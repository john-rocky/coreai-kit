# InfoExtract — zero-shot entity extraction and PII redaction on device

Pull structured entities out of free text without training a model for your labels: you pass the
label set at call time. Names, emails, phone numbers, card numbers, addresses — or any labels you
invent, since the model is zero-shot. Nothing leaves the device, which is the point when the text
you are extracting from is exactly the text you cannot send anywhere.

The model is [GLiNER2-PII](https://john-rocky.github.io/coreai-model-zoo/models/gliner2-pii/).
The whole ML surface is one call:

```swift
let extractor = try await InformationExtractor(catalog: "gliner2-pii")
let found = try await extractor.extract(from: text, entities: ["email", "phone number"])
```

For the common case there is a one-liner that redacts instead of returning spans:

```swift
import CoreAIOps
let clean = try await CoreAI.redact(text)
```

## Run

```bash
swift run infoextract-cli --text "email me at a@b.com" --labels email
swift run infoextract-cli --gate       # reproduce the demo PII suite

xcodegen generate                      # app
open InfoExtract.xcodeproj
```

Omit `--bundle` and the model downloads from the Hugging Face Hub on first use. Pass a directory
holding the `.aimodel` plus `tokenizer/` and `extractor.json` to run a local build instead.

## Notes

- Progress goes to stderr, results to stdout, so the CLI's output stays machine-checkable —
  useful if an agent or a script is asserting on it.
- The label set is a runtime argument, not a property of the model. Adding a category costs a
  string, not a retrain — but recall varies by how well the label names describe what you want,
  so check yours against your own text rather than assuming the defaults transfer.
