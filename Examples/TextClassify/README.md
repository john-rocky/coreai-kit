# TextClassify — your messages, sorted into categories you name

`gliner2.5-decide` (`mlboydaisuke/GLiNER2.5-Decide-CoreAI`) on Core AI gives every message you paste a category you
name, an urgency and a sentiment: 38.4 to 38.6 ms per message (median, two runs) on an iPhone 18 Pro, measured
2026-10-10.

Paste your messages, import a text file, or take the sample inbox. Name 2 to 12 categories. One tap asks the model
three questions about every message, and one run of the model answers all three. **By category** then lists each
category with its most urgent messages first. Nothing leaves the device.

The model is [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (fastino, Apache-2.0) exported to
Core AI. Labels are named when you call it, so your categories need no training. The whole ML surface is one call:

```swift
let classifier = try await TextClassifier()   // the catalog's gliner2.5-decide, downloaded once
let answers = try await classifier.classify(text, tasks: [
    ClassificationTask("intent", labels: ["work", "family", "bills", "travel", "shopping"]),
    ClassificationTask("aspects", labels: ["battery", "keyboard", "screen"], multiLabel: true, threshold: 0.4),
])
answers["intent"]?.labels          // ["bills"]
answers["aspects"]?.probabilities  // every label with its probability
```

One question has a shorter form:

```swift
let (label, probability) = try await classifier.classify(text, labels: ["spam", "ham"])
```

A task can also carry a `prompt` (the question to answer about the text) and `descriptions`
(what each label means).

## Where the messages come from

Three buttons at the top pick the source.

- **Paste**. One message a line. Messages that run over several lines, such as pasted emails, need a blank line
  between them; each block is then one message. A line with no letter or digit (a rule like `-----`) is skipped,
  and the sheet counts it. A paste holds at most 2,000 messages; a longer one is refused whole.
- **Import file**. A `.txt` or `.md` file is read like a paste. A `.csv` file gives one message a row: the quotes
  come off, and a row's fields are joined with ", ". The file must be UTF-8. Any other encoding is refused with the
  reason; nothing is guessed.
- **Sample inbox**. 100 messages from the generator (seed 7), the inbox `textclassify-cli --inbox 100` sorts. It is
  generated, not collected (`Sources/InboxCore/Inbox.swift`): an invented shop and app subscription, plain item
  names, first names only, no addresses or contact details.

With nothing to sort, the screen says why and offers the three sources.

## Categories

The **Categories** field takes 2 to 12 names, separated by commas. Each is under 40 characters, and no name may
appear twice (case does not count). The default is a support inbox's eight intents.

Your names replace those eight as the labels of the first question, which the model reads as `intent` either way.
Urgency (four levels, each with a description) and sentiment (three) are asked as shipped, so the urgency order
means the same whatever the categories are. The bundle takes 32 labels a call; twelve categories and the seven
urgency and sentiment labels make 19. Editing the field after a sort clears the answers.

## The screen

1. **Sort inbox** asks every message its three questions, in order. In **Inbox**, each row gets three chips as its
   answers arrive: the category, the urgency and the sentiment. The clock, the median and the rate on screen are
   the app's own measurements.
2. **By category** lists each category with its count. Under it come its messages: critical first, then high,
   normal and low, each with its sentiment. It fills as the answers arrive.
3. The bars under the list count each category, and the line under them counts the critical and high messages.

A message longer than the 512-token graph is cut at a word boundary from the end, and its row says **truncated**.
The model does not read the cut words, so an urgent last sentence can be missed.

## Measured

2026-10-10, Release builds on the GPU, the fp16 bundle at revision `820d4e9`: from the catalog on the Mac, copied into
the app's Documents on the iPhone. Each run is a fresh launch: load, one warm-up input through each graph, then the
press. The times are the app's own.

| Run | Device | Messages | Total | Median per message | p90 | Load |
|---|---|---|---|---|---|---|
| Sample inbox | M4 Max, macOS 27.0 (26A428) | 100 | 3.05 s | 30.1 ms | 30.9 ms | 0.59 s |
| The same 100 pasted, run 1 | M4 Max | 100 | 3.33 s | 30.0 ms | 30.8 ms | 0.91 s |
| The same 100 pasted, run 2 | M4 Max | 100 | 3.43 s | 31.0 ms | 32.0 ms | 0.91 s |
| My 10 messages, my 5 categories | M4 Max | 10 | 0.34 s | 30.9 ms | 32.6 ms | 0.91 s |
| The same 100 pasted | iPhone 18 Pro, iOS 27.2 (24B5099f) | 100 | 3.93 s | 38.6 ms | 38.8 ms | 0.70 s |
| The same 100 pasted, By category | iPhone 18 Pro | 100 | 3.90 s | 38.4 ms | 38.7 ms | 0.52 s |
| My 10 messages, my 5 categories | iPhone 18 Pro | 10 | 0.44 s | 39.5 ms | 41.8 ms | 0.53 s |

The four Mac runs share one 60-second measurement window, opened after the GPU had sat at 5 % or less for 60 s.
macOS's photo analysis (`mediaanalysisd`) used two to three CPU cores during it. A pasted run's total includes
0.29 s before the first message, while the sheet closes and the list is drawn; the time per message is the same.

In both pasted runs, every message got the same three answers as in the sample run, 100 of 100, with the same
probabilities to four decimals. The sample run's categories: 17 cancel subscription, 17 technical issue,
14 order status, 14 refund request, 13 general feedback, 12 feature request, 7 billing question, 6 account access.

My own test: five categories (`work, family, bills, travel, shopping`) and ten messages, two for each, with the
category each should get written down before the first run. All ten got that category, on the Mac and on the iPhone.

On the iPhone, the first launch after the install loaded in 5.98 s: each graph was compiled for the phone on its first
run (3.18 s and 2.68 s). Later launches loaded in 0.52 to 0.70 s. Each phone run ended within 4 s, before the 20 s
mark where the 2026-09-26 runs began to slow, and the thermal state stayed nominal. The phone gave all 100 messages the same three answers as the Mac; the
chosen labels' probabilities differ by at most 0.0007.

Measured 2026-09-26 (iOS 27.0, before the paste and file inputs), seed 7: 1,000 messages in 40.9 s on an iPhone 18
Pro, 39.2 ms median per message, all on the 256-token graph; 30.1 s on an M4 Max (30.0 ms). The phone held about
38 ms per message for the first 20 s of a run and slowed from there (1.25–1.5× by 41 s over five runs, with the
thermal state still nominal), so a longer inbox takes more than its share: 2,000 messages took 128.7 s.

## Run

```bash
cd App && open TextClassify.xcodeproj   # pick your team in Signing & Capabilities, then Run
cd App && xcodegen generate             # after editing project.yml
xcodebuild -project App/TextClassify.xcodeproj -scheme TextClassify -destination 'platform=macOS' DEVELOPMENT_TEAM=<team> build
```

The team goes to `xcodebuild` as a build setting: the project holds `${DEVELOPMENT_TEAM}`, which an exported
environment variable does not fill at build time.

The model comes from the catalog on first launch (`gliner2.5-decide`, 1.76 GB on a Mac, cached), or from a folder
already on the device: on an iPhone, a copy of the bundle at `Documents/gliner25-decide/` (put there with
`xcrun devicectl device copy to`); on a Mac, `-bundle <dir>`. The download, the load and a sort hold a
user-initiated activity (`ProcessInfo.beginActivity`). Without it, macOS slows the app while its window sits behind
others (App Nap): a first download on an M4 Max fell from 7–10 MB/s to 0.1–0.2 MB/s after about 30 s.

```bash
swift run textclassify-cli --text "Can I get that charge refunded?" \
    --task intent=order_status,refund_request,cancel_subscription
swift run textclassify-cli --text "Battery dies before lunch, but the screen is great." \
    --task aspects=battery,keyboard,screen --multi --threshold 0.4 --task sentiment=positive,negative,mixed
swift run textclassify-cli --bundle <dir> --readme readme21.json     # the model card's 21 examples
swift run textclassify-cli --bundle <dir> --gate <oracle fixtures> --pygpu <gate_s256_gpu.json> <gate_s512_gpu.json>
swift run -c release textclassify-cli --inbox 1000 --seed 7   # the sample inbox, headless, one JSON line
swift run textclassify-cli --inbox 100 --seed 7 --dump        # the sample inbox as lines, in the Paste form
swift test                                                    # how a paste, a file and the categories are read
```

Without `--bundle` the CLI downloads the catalog's `gliner2.5-decide` on first use and caches it
(`TextClassifier()` does the same in an app). `--bundle` takes a local export instead, the directory
the zoo's GLiNER2.5-Decide export writes: `classifier.json`, a `tokenizer/` folder, and one graph
per sequence length (256 and 512 tokens). `--multi`, `--threshold`, `--prompt` and `--describe label=text` apply to the
`--task` before them. `--inbox` sorts the sample inbox through the app's own `InboxSorter` (`--jsonl <path>` writes
every answer).

## Unattended runs

For a recording or a scripted check, the app presses its own buttons:

```
TextClassify -autoplay 1 -source paste|file|sample [-text <file>] [-categories "work, family, bills"] \
    [-view inbox|category] -count 100 -seed 7 -delay 3 -log 1 [-out <dir>] [-trigger <file>] [-bundle <dir>]
```

`-source paste -text <file>` opens the Paste sheet holding the file's text and presses the sheet's Sort, so the file
gets the reading a paste gets, refusals included. `-source file -text <file>` goes through Import file's reading.
`-categories` is the field's text, and `-view` the view the app opens on. With `-log 1` the status line goes to
`inbox-autoplay.log` as it changes, and every finished run writes `inbox-result-<epoch>.json`: the CLI's summary
plus `source`, `categories`, `messages_skipped`, every message's three answers with their probabilities (never its
text), the per-message series and where it ran. Both files go to `-out`, else Documents. On a Mac, pass `-out`:
Documents asks for folder access and may sync to iCloud. On an iPhone, `devicectl device copy to` puts a `-text`
file in Documents and `devicectl device copy from` reads the results back (put `--` before the app's arguments in
`devicectl device process launch`, or it reads `-log` as its own option).

## Notes

- Progress goes to stderr, results to stdout (one JSON line, in gliner2's
  `classify_text(..., include_confidence=True)` shape), so an agent or a script can assert on it.
- The tasks are also part of the model's input, so asking a second question moves the first one's probabilities a
  little: classify with the task set you will use.
- The classifier runs the smallest graph the text fits.
- `--gate` is what "matches gliner2" means here: for every case of the zoo's oracle fixtures it
  compares the collated input (token ids, pieces, the [P]/[L] positions and the shape) with gliner2
  2.0.0's, then runs the graph and compares each decision with gliner2's fp32 decision and the
  logits with the oracle's (and, with `--pygpu`, with the zoo's Python run of the same bundle).
