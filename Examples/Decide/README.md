# Decide — typed decisions on device

A state and a typed question in, an answer with its probability out — nothing generated. The
model scores one prompt and the answer is read at its answer slot: **choice** (which of these
options), **score** (where on this scale), **noul** (yes or no, as P(yes)). One line:

```swift
let a = try await CoreAI.decide(
    text,
    ["reply":  .noul("Does the speaker want a response from the assistant?"),
     "topic":  .choice("What is the message about?", ["billing", "delivery", "other"]),
     "urgent": .score("How urgent is it?", levels: ["can wait", "this week", "today"])])
a["reply"]?.noul      // P(yes)
a["topic"]?.choice    // "delivery"
a["urgent"]?.score    // expected level, 0…2
```

Five screens, one loaded model, each one a whole use: one action, the complete result. The
same sources build for the Mac and for the iPhone:

| Screen | What you do → what you get | Shape |
|---|---|---|
| **Autofill** | Copy an email or a message (⌘C in Mail is enough; Paste on the iPhone) → every field of a checkout form (HTML, in a web view) fills at once. The text is prefilled once; one question per field picks the line that holds it — "which line has the street of the address the order should be shipped to?" — as a choice among the text's lines, then the line is trimmed to what the field takes. A copied single piece goes to its one field; a secret key is refused. | choice, 7 per email |
| **Checklist** | Open a contract (text, Markdown, PDF, or a photo of one) → a list of verdicts to your questions — `noul:` / `choice:` / `score:` lines you edit. Read once, answered N times. | noul + choice + score |
| **Sorter** | Open a folder → what needs you first (a payment, a reply by a date), then everything filed by folder; **Move files** does it. Every file read once, two decisions each. | choice + choice |
| **Search** | Query × passages: one decision per passage, ranked by its probability — a reranker made of a chat model, no index. | noul or score |
| **Speech gate** | Record, split into utterances, transcribe each on device (Apple's recognizer, no download), and ask one question per utterance: is this a request for the assistant, or a remark? Only the ones that pass would go to a language model. | choice |

The two Shortcuts actions (“Ask yes/no”, “Classify text”) run the same decisions on any text
without opening the app.

## Run

```bash
# GUI (iPhone or Mac)
xcodegen generate
open Decide.xcodeproj

# headless (macOS), the same function the GUI calls
swift run -c release decide-cli ask --state "The order arrived with the box crushed." \
    --noul "Does the customer want a replacement?" \
    --choice "What is it about?|billing|damaged delivery|how-to" \
    --score "How urgent?|can wait|this week|today"
swift run -c release decide-cli bench --model minicpm5-2b        # shared vs direct prefill
swift run -c release decide-cli --list-models

# one decision per line of stdin — a semantic grep
gh issue list --json title -q '.[].title' | swift run -c release decide-cli filter \
    --noul "Is this a crash report?"

# a decision model's own fixture: token ids, answer slots and probabilities, row by row
swift run -c release decide-cli parity --model decider-0.8b \
    --fixture fixtures-decider-0.8b.json --states states.json
```

Hands-off, for a recording or a smoke run: `Decide.app/Contents/MacOS/Decide -autoplay
sorter -model minicpm5-2b` opens that tab, loads the model and presses the screen's own
sample button (`-trigger <path>` waits for that file first; `-delay <s>` after Ready).

## Measured

Apple M4 Max, macOS 27.0 (26A428), Release, sequential engine, MiniCPM5 2B int8
(`minicpm5-2b`) unless noted; other GPU work was running on the machine at the time, so the
timings are an upper bound, not a quiet-machine figure. Dates are when the numbers were
taken.

**Checklist** (2026-09-22): the sample lease, 565 tokens, twelve questions. Read once in
316 ms, then twelve answers in 532 ms. All twelve read the lease correctly:
no subletting without consent, an early-termination fee, pets allowed, no rent increase
during the term, the landlord repairs the dishwasher, rent by bank transfer, a 12-month term,
60 days' notice, smoking not allowed. (Asked as a yes/no, "is smoking allowed?" came back
*yes* at 0.82; asked as a choice — allowed / not allowed / not mentioned — the same model says
not allowed at 1.00. The sample asks it as a choice.)

**Sorter** (2026-09-22): the sample folder, twelve invented documents, four named folders.
Twenty-four decisions in 1,558 ms. All twelve land in the intended folder (eleven at
confidence 1.00, one at 0.63); five are listed under *Needs you* — the invoice and the payment
reminder (a payment), the lease renewal, the NDA and the neighbour's letter (a reply,
signature or decision by a date) — and the three manuals, the paid receipt, the warranty, the
thank-you card and the note are filed without a flag.

**Clipboard, watching** (2026-09-22): eight copies, two decisions each, 202–211 ms per copy
(269 ms for the first, which pays the engine's specialization at a new length). All eight
kinds are named as intended — an address, a secret key, a phone number, a link, a date, an
email address, ordinary prose, a tracking number — and against "a shipping address" the
address alone is "paste as-is", the phone number "part of what you need", the rest "not what
you need".

**Autofill** (2026-09-22): the sample email (12 lines, with an old address as a decoy), copied
once. Prefilled once, then seven decisions — one per field, plus one for where the address
ends — in 1,370 ms; all six fields filled, the new address chosen over the old one
(street 0.64, its last line against the three lines after it).

**Speech gate, sample** (2026-09-22): eight utterances, median 64 ms per decision. Five pass:
the four requests and "Hold on, let me find my keys." (0.89); the three remarks are held back
at 0.01–0.17.

**Search, sample** (2026-09-22): six passages in 487 ms, median 65 ms each; the returns
passage ranks first at P(yes) 1.00.

**Per decision, shared vs from scratch** (2026-09-21; a 150-token state and eight questions of
40–60 tokens each; median of 3 runs):

| Model (catalog id) | ms per decision, state shared | ms per decision, every prompt from scratch | prefill + 8 decisions on one state |
|---|---:|---:|---:|
| MiniCPM5 2B int8 (`minicpm5-2b`) | 64.5 | 108.8 | 541 ms vs 1013 ms |
| MiniCPM5 1B int8 (`minicpm5-1b`) | 24.8 | 48.2 | 273 ms vs 468 ms |
| Qwen3 0.6B 4-bit (`qwen3-0.6b`) | 15.8 | 33.6 | 163 ms vs 308 ms |

**Shared** = the state is prefilled once and each question rewinds the engine to it
(`Decision.Timing.reusedTokens` = 150 here); **from scratch** = the whole prompt every time.

Agreement with the published bf16 readout of the same models on the same 144 authored rows
(SemIf's `authored144` fixture, its `direct` rendering byte for byte, its `evaluate.py` metric):

| Model | Prompt tokens identical | Argmax agreement | max / mean \|Δp\| | Mean family balanced accuracy (kit) | Published bf16 |
|---|---:|---:|---:|---:|---:|
| MiniCPM5 2B int8 | 144/144 | 141/144 | 0.183 / 0.020 | 0.681 | 0.686 |
| MiniCPM5 1B int8 | — | — | — | 0.513 | — |
| Qwen3 0.6B 4-bit | 144/144 | 59/144 | 1.000 / 0.580 | 0.333 | 0.440 |

So `minicpm5-2b` is the default: its int8 decisions track the fp32 model. The official 4-bit
Qwen3 0.6B bundle does **not** — it answers the last option far too often — so use it for its
speed only where a wrong decision is cheap.

**A model trained for decisions.** `decider-0.8b` (catalog kind `decision`, `Decision.Format.decider`)
was fine-tuned to answer typed questions at an answer slot; the kit renders its own prompt
form and reads it at its card's temperature. On the model's 44-row fixture (`decide-cli
parity`, 2026-09-22, int8 bundle): the 43 rows the kit can list (the 255-option row is beyond
its 16) are token-identical, slot-identical and argmax-identical to the author's fp32 readout,
max |Δp| 0.0088, mean 0.0009. It ships as a decode-only graph on a recurrent hybrid, so every
row re-prefills its whole prompt one token at a time: median 343 ms per fixture question
here, about 490 ms per clipboard kind and 710 ms per sorter need — five to seven times
`minicpm5-2b`'s shared-prefix figures. On the two demo questions it was not more accurate
than the chat model (kinds 5/8, needs 11/12). Its place is where the probability must mean
what the model's own API would report; the screens default to `minicpm5-2b`.

**What the shape does to a small model's answer.** Every question above was tried in
several shapes before it went in (the CLI's `filter` is how). With MiniCPM5 2B, a yes/no on
a short text leans *yes*: "is this what the purpose needs?" says yes to a phone number, a
URL and an API key (P(yes) ≥ 0.7), "does this document need something from you?" says yes to
a manual, "is the speaker asking the assistant to do something?" says yes to three of four
remarks. The same model, asked to *choose* — "which one is this text?" with an article per
option, "what does this document need from the reader?" with "nothing: a receipt, a manual,
terms to keep" as the first option, "a request for the assistant" against "small talk, a
remark, or talking to someone else" — separates them; and "is this text a shipping address?"
on a no / partly / completely scale puts the address alone at completely. Prefer a choice
with a named "none" option, or a scale, over a bare yes/no when the state is short. The
yes/no shape still reads well on a long state (the checklist's lease).

No iPhone numbers yet; the engine path is the same.

## Where the code is

- `Sources/QuickStart.swift` — the take-home: one typed function, no UI. The GUI and the CLI
  both call it.
- `CLI/main.swift` — argument shell over that function, plus `bench`, `oracle`, `parity`
  and `filter` (the numbers above).
- `Sources/DecideRuntime.swift` — the one loaded `TypedDecisions` the screens share.
- `Sources/Form*.swift` (Autofill; `FormPage.html` is the checkout page it fills through one
  JavaScript call), `Checklist*.swift`, `Sorter*.swift`, `Search*.swift`, `SpeechGate*.swift`
  — the five screens; `Intents.swift` — the Shortcuts actions;
  `DocumentText.swift` — a file as text (plain, Markdown, PDF, image via Vision);
  `Autoplay.swift` — the hands-off runner.

## Integration checklist

- SPM: `github.com/john-rocky/coreai-kit` → product `CoreAIOps` (or `CoreAIKit` for
  `TypedDecisions` alone)
- Info.plist: `NSMicrophoneUsageDescription` and `NSSpeechRecognitionUsageDescription` only
  for the speech gate; the other screens need nothing (the file importers are system pickers)
- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit` for the 2.7 GB
  MiniCPM5 2B bundle
- First run downloads the model → cached in Application Support (progress callback)
- Measure in Release — Debug is ~3× slower on per-token host work
