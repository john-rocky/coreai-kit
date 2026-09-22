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

Ten screens, one loaded model, each one a whole use: one action, the complete result. The
same sources build for the Mac and for the iPhone. The last five are the shapes the
most-viewed System One posts of September 2026 use — a game driven by the model, bulk
classification of rows, a natural-language permission gate for a coding agent, context
compression by relevance, as-you-type reading — with the model on the device:

| Screen | What you do → what you get | Shape |
|---|---|---|
| **Autofill** | Copy an email or a message (⌘C in Mail is enough; Paste on the iPhone) → every field of a checkout form (HTML, in a web view) fills at once. The text is prefilled once; one question per field picks the line that holds it — "which line has the street of the address the order should be shipped to?" — as a choice among the text's lines, then the line is trimmed to what the field takes. A copied single piece goes to its one field; a secret key is refused. | choice, 7 per email |
| **Checklist** | Open a contract (text, Markdown, PDF, or a photo of one) → a list of verdicts to your questions — `noul:` / `choice:` / `score:` lines you edit. Read once, answered N times. | noul + choice + score |
| **Sorter** | Open a folder → what needs you first (a payment, a reply by a date), then everything filed by folder; **Move files** does it. Every file read once, two decisions each. | choice + choice |
| **Search** | Query × passages: one decision per passage, ranked by its probability — a reranker made of a chat model, no index. | noul or score |
| **Speech gate** | Record, split into utterances, transcribe each on device (Apple's recognizer, no download), and ask one question per utterance: is this a request for the assistant, or a remark? Only the ones that pass would go to a language model. | choice |
| **Drive** | Press Drive → the model drives a car down a three-lane road. Every tick it reads the lanes it can reach — "left lane: rock 1 ahead", "middle lane: clear" — and picks one; the road advances; rocks come. The tick rate is the decision rate. | choice, one per tick |
| **Columns** | Open a CSV (or paste one text per line), write the columns as questions → every row is answered: read once, one decision per column; sort by any column, save the CSV with the new columns. | choice / score / noul per column |
| **Command guard** | Paste the commands an agent wants to run, with a policy in plain words → each one comes back run / ask the user first / refuse, the refused ones on top. `hooks/claude-code-guard.sh` runs the same decision as a Claude Code PreToolUse hook. | choice, one per command |
| **Context** | An agent transcript's tool results, scored against the question being answered → the unrelated ones drop out of the context; the header says how many tokens went. | score, one per tool result |
| **Typing** | Type → at every pause three chips read the text so far: its tone, what you are doing, the emoji that fits (tap to insert). | score + choice + choice per pause |

The two Shortcuts actions (“Ask yes/no”, “Classify text”) run the same decisions on any text
without opening the app.

## The same endpoint your client already speaks

`decide-cli serve` (or `systemone serve`, the same server as one Homebrew-installed binary:
`brew install john-rocky/tap/systemone`) puts the loaded model behind a `/v1/systemone`
endpoint on this machine, in
the request and answer forms of the hosted System One API — `state` (a string, or structured
data), `model`, `questions` keyed by your ids with `type` / `instructions` / `criteria`;
`answers` back with `choice` and every option's probability, `score` with its legend, `noul` as
the probability the statement holds, `confidence`, `usage`. A client written for the hosted
endpoint is pointed at this one by its base URL and nothing else changes:

```bash
swift run -c release decide-cli serve --model minicpm5-2b          # http://127.0.0.1:8090/v1/systemone
clients/systemone.sh                                               # one request with curl
python3 clients/systemone.py                                       # the same from Python, standard library only
SYSTEM_ONE_BASE_URL=http://127.0.0.1:8090 python3 your_client.py   # a client library that takes a base URL
```

```json
{"state": "Help! My payouts have been failing for 3 days.", "model": "minicpm5-2b",
 "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                             "criteria": {"true": "explicitly time-sensitive", "false": "no urgency expressed"}},
               "queue": {"type": "choice", "instructions": "Which queue should handle this?",
                         "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}}}}
```
```json
{"model": "minicpm5-2b",
 "answers": {"is_urgent": {"type": "noul", "noul": 0.9274, "confidence": 0.9274},
             "queue": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.5886, "technical": 0.4109, "other": 0.0004}, "confidence": 0.3804}},
 "usage": {"input_tokens": 239, "output_tokens": 2}, "timing_ms": 103.5733}
```

The state is prefilled once per request and every question rewinds to it; that request took
204 ms end to end on the Mac (M4 Max, 2026-09-23), four questions on a 69-token state 436 ms.
A third-party client library written for the hosted endpoint (`system-one` 0.1.0 on PyPI,
`HTTPConfig(base_url=…)`) got its typed answers back from this server unchanged, 179 ms round
trip for three questions.
A structured `state` or `instructions` (an object or an array) is serialized the way Python's
`json.dumps(…, ensure_ascii=False)` writes it, key order kept, so the model reads the bytes a
Python client would have sent (`JSONValue` in the kit). `confidence` is 1 − normalised entropy
of the distribution for a choice or a score and max(p, 1 − p) for a noul; probabilities are
rounded to four decimals. One declared difference from the hosted API: a choice lists at most
16 options here (the answer slots are single letters), so a longer list comes back as a 422
that says so. `GET /v1/models` names the loaded model, `GET /health` answers `ok`, CORS is open
so a page or a browser extension can call it, and `--host 0.0.0.0` serves the local network
(a phone on the same Wi-Fi, another machine). The codec is public API (`SystemOne.request(from:)`,
`SystemOne.response(model:answers:)`) for an app that wants to accept or emit the form itself.

**The same decision as a hook.** `hooks/claude-code-guard.sh` is a Claude Code `PreToolUse`
hook: every Bash command the agent is about to run goes through `decide-cli` under the policy
at the top of the script, and "ask the user first" / "refuse" become the hook's
`permissionDecision`. Build `decide-cli` once, point the hook at it in `settings.json`
(the script's header shows the entry), and the gate runs on your machine — each call loads the
model (a few seconds with the weights cached) and decides in about 70 ms.

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
swift run -c release decide-cli parity --model openthai-systemone \
    --fixture fixtures-openthai-systemone.json      # JSON states render themselves; --bundle <dir> reads an unpublished port
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

**Drive** (2026-09-23): the recorded run, 17 seconds hands-off, 52 ticks, 52 decisions,
median 33 ms each, 22 rocks passed, no crash; the screen paces itself at 0.3 s per tick so
the Mac stays watchable. The shape is what makes it work: asked "which move?" with stay /
move left / move right, the same model stayed in its lane into a rock four times in twelve
states; asked "which lane?" with each reachable lane described by what lies ahead in it, it
never picks a rock lane when a clear one is offered (12/12) and keeps its lane while it is
clear. What it cannot do is compare two bad lanes — offered "rock 1 ahead" against "rock 2
ahead" it took the nearer rock and crashed at row 30 of an earlier take — so the road never
puts rocks in different lanes on consecutive rows: at most one reachable lane is ever blocked
within the two rows the descriptions cover, and a clear option always exists.

**Columns** (2026-09-23): the sample CSV, twenty support tickets, three columns — topic
(six-way choice), what the customer wants (six-way choice), mood (a three-level score) —
sixty decisions in 3,930 ms, about 200 ms per row including its prefill. Topic and wants
read as intended on 18–19 of 20 rows (a discount-code complaint lands in "none of these"
rather than billing; "cancel my subscription" reads as wanting a refund); the mood scale
puts the two thank-you notes at happy and the complaints at upset. There is no urgency
column: asked how soon a short ticket needs an answer, in any shape tried, the model says
"today" for nearly every row.

**Command guard** (2026-09-23): the sample log, fourteen commands, in 1,166 ms — six run
(status, test, cat, an in-project sed, a new branch, an install), six ask first (a force
push, a delete outside the project, a production `DROP TABLE`, a production namespace
delete, `rm -rf node_modules`, `sudo rm -rf /`), two refused (a download piped into a
shell, a private key posted to a paste site). Nothing the policy refuses or holds got
through; the two conservative verdicts are the `node_modules` delete (ask, not run) and the
disk wipe (ask, not refuse). With the three options named alone the same model asked about
nine of the fourteen; naming each option with the policy's own words for it is what gives
the split above.

**Context** (2026-09-23): the sample transcript, thirteen tool results, 1,832 tokens by the
model's tokenizer → seven kept, 958 tokens, thirteen decisions in 847 ms. The four results
the question depends on (the session code, the failing test, the grep, the changelog line)
score 0.97–1.23 on the none / some / most scale; the weather, the pull-request list, the
README, the release script and the pagination code score 0.00–0.71 and drop; the file
listing, the git log and the API doc sit at 0.85–1.04 and stay. The line is 0.8.

**Typing** (2026-09-23): "just got the job offer!! dinner tonight to celebrate?", typed by
the hands-off run — positive, sharing news, 😀; three decisions in 279 ms at the last pause.

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

**A decision model with a head of its own.** `openthai-systemone` (catalog kind `decision`,
`Decision.Format.slot`; iApp's OpenThai-SystemOne, Thai + English, Qwen3.5-0.8B tower,
Apache-2.0) replaces the LM head with a 256-way slot head read at a `<|ts_answer|>` control
token: option i is slot i, the last slot means "none of these" (`Decision.Answer.abstain`), and
a choice may list up to 255 options (`TypedDecisions.maxOptions`), not the 16 of the letter
readouts. The bundle declares the head and its temperature per question type in its
`metadata.json`; the kit renders the author's control-token layout, one question per row. On
the author's 50-row fixture (`decide-cli parity`, 2026-09-23, Thai, English and JSON states,
rows of 2–16, 40 and 255 options): tokens, slots and argmax identical on 50/50 for both the
int8 bundle (max |Δp| 0.0226 on one two-option row, mean 0.0009) and the fp16 reference
(0.0051 / 0.0003). On SemIf's 144 English authored rows, its own form, the same evaluator as
the table above: mean family balanced accuracy 0.725 (int8). Speed on the Mac: a 137-token
Thai ticket's first question 170 ms, the two that share its prefix 40 and 65 ms; over the
fixture's 50 rows, two to three questions per state, 354 ms median per question (the graph
prefills one token at a time, and a question on a new state pays for the whole state), the
255-option row 7.2 s. Two things to know: its author's API lays several questions in one
sequence and answers them in one pass, and those answers can differ from the one-question
rows the kit sends (up to 0.375 on the fixture's requests) — the kit's rows equal the
author's single-question API exactly; and Thai is cut the way the reference tokenizer cuts it
(at every combining mark), which the Swift tokenizer alone does not do — `SlotPrompt.Encoder`
does the cutting, and 23 of the 50 rows differed before it did.

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

**iPhone 17 Pro** (iOS 27.0, airplane mode, the same app and the same model, 2026-09-23):
the model loads in 22 s the first time after install and 1.4–4.1 s once the system has it
cached; the lease reads in 508 ms and its twelve answers take 1,134 ms (the first run after
install: 1,740 / 1,589 ms); the email's seven decisions take 2,001 ms; the folder's
twenty-four take 2,579 ms. The clips in coreai-assets `kit/decide/*-iphone.mp4` show these
runs; the first-run figures were read back over `devicectl` from the app's `-log 1` file.

**iPhone 17 Pro, the five newer screens** (iOS 27.0, airplane mode, the same app and the
same int8 bundle, the recorded runs, 2026-09-23): Drive, 17 seconds hands-off: 53 ticks, 53
decisions, median 64 ms each, 19 rocks passed, no crash. Columns: the twenty tickets × three
columns, 60 decisions in 7,420 ms, every cell the same as on the Mac (20/20). Command guard:
fourteen commands in 1,906 ms, the same 6 / 6 / 2 verdicts at the same confidences. Context:
the thirteen tool results in 1,454 ms, the same seven kept, scores within 0.01 of the Mac's.
Typing: three decisions in 484 ms at the last pause. The model loads in 1.5–5.5 s from the
cache; the hands-off `-log 1` runs before the recording gave the same figures within a few
percent (Drive 80 ticks in 25 s at a 63 ms median). The same bundle on both machines gives
the same decisions; the phone takes about twice the Mac's time per decision. Clips:
coreai-assets `kit/decide/{drive,columns,guard,context,typing}-iphone.mp4`.

## Where the code is

- `Sources/QuickStart.swift` — the take-home: one typed function, no UI. The GUI and the CLI
  both call it.
- `CLI/main.swift` — argument shell over that function, plus `bench`, `oracle`, `parity`
  and `filter` (the numbers above) and `serve`, a shell over the kit's `SystemOneServer`
  (`Sources/CoreAIKit/Decide/SystemOneServer.swift`); `clients/` — a curl and a Python
  request to it.
- `Sources/DecideRuntime.swift` — the one loaded `TypedDecisions` the screens share.
- `Sources/Form*.swift` (Autofill; `FormPage.html` is the checkout page it fills through one
  JavaScript call), `Checklist*.swift`, `Sorter*.swift`, `Search*.swift`, `SpeechGate*.swift`,
  `Drive*.swift`, `Columns*.swift`, `Guard*.swift`, `Context*.swift`, `Typing*.swift` — the
  ten screens; `Intents.swift` — the Shortcuts actions; `hooks/claude-code-guard.sh` — the
  guard as a Claude Code hook;
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
