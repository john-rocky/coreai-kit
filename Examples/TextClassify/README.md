# TextClassify — zero-shot text classification on device

Sort text into labels you name at call time: an intent, a queue, a severity, a yes/no — several
questions about one text in a single forward. No training for your label set, and nothing leaves
the device.

The model is [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (fastino,
Apache-2.0) exported to Core AI. The whole ML surface is one call:

```swift
let classifier = try await TextClassifier()   // the catalog's gliner2.5-decide, downloaded once
let answers = try await classifier.classify(text, tasks: [
    ClassificationTask("intent", labels: ["order_status", "refund_request", "cancel_subscription"]),
    ClassificationTask("aspects", labels: ["battery", "keyboard", "screen"], multiLabel: true, threshold: 0.4),
])
answers["intent"]?.labels          // ["refund_request"]
answers["aspects"]?.probabilities  // every label with its probability
```

One question has a shorter form:

```swift
let (label, probability) = try await classifier.classify(text, labels: ["spam", "ham"])
```

A task can also carry a `prompt` (the question to answer about the text) and `descriptions`
(what each label means).

## Run

```bash
swift run textclassify-cli --text "Can I get that charge refunded?" \
    --task intent=order_status,refund_request,cancel_subscription
swift run textclassify-cli --text "Battery dies before lunch, but the screen is great." \
    --task aspects=battery,keyboard,screen --multi --threshold 0.4 --task sentiment=positive,negative,mixed
swift run textclassify-cli --bundle <dir> --readme readme21.json     # the model card's 21 examples
swift run textclassify-cli --bundle <dir> --gate <oracle fixtures> --pygpu <gate_s256_gpu.json> <gate_s512_gpu.json>
```

Without `--bundle` the CLI downloads the catalog's `gliner2.5-decide` on first use and caches it
(`TextClassifier()` does the same in an app). `--bundle` takes a local export instead, the directory
the zoo's GLiNER2.5-Decide export writes: `classifier.json`, a `tokenizer/` folder, and one graph
per sequence length (256 and 512 tokens). `--multi`, `--threshold`, `--prompt` and `--describe label=text` apply to the
`--task` before them.

## Notes

- Progress goes to stderr, results to stdout (one JSON line, in gliner2's
  `classify_text(..., include_confidence=True)` shape), so an agent or a script can assert on it.
- The labels of all tasks in one call share the bundle's label slots (32). The tasks are also part
  of the model's input, so asking a second question moves the first one's probabilities a little:
  classify with the task set you will use.
- The classifier runs the smallest graph the text fits. Text longer than the largest one is cut at a
  word boundary from the end, and every result says `truncated`.
- `--gate` is what "matches gliner2" means here: for every case of the zoo's oracle fixtures it
  compares the collated input (token ids, pieces, the [P]/[L] positions and the shape) with gliner2
  2.0.0's, then runs the graph and compares each decision with gliner2's fp32 decision and the
  logits with the oracle's (and, with `--pygpu`, with the zoo's Python run of the same bundle).
