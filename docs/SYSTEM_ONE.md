# System One, on device

A System One call is a typed decision: a text (the state), a question with a fixed set of
answers, and each answer's probability back. Nothing is generated. Three shapes cover it —
**choice** (which of these), **score** (where on this ordered scale), **noul** (does this hold,
as a probability) — and a request asks several questions of one state at once.

This page is the map of what CoreAIKit does with that on a Mac or an iPhone, with the numbers
and where they came from.

## The call

Swift, one line per shape (`import CoreAIOps`):

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

The same call in the hosted API's own forms — a request that arrived as JSON, a response to
hand back as JSON — is `CoreAI.systemOne`, the typed answers beside the wire object:

```swift
let r = try await CoreAI.systemOne(json: body)   // the bytes a client sent: state, questions, model
r["topic"]?.choice                               // typed, by the request's own key
r.dumps()                                        // the reply, byte for byte what `systemone serve` returns
```

Anything else, over HTTP, in the hosted API's own forms:

```bash
brew install john-rocky/tap/systemone && systemone serve         # http://127.0.0.1:8090/v1/systemone
brew services start systemone                                     # the same, kept running by launchd
curl -s http://127.0.0.1:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.", "model": "minicpm5-2b",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                "queue": {"type": "choice", "instructions": "Which queue?",
                          "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}}}}'
```

`systemone` is one signed, notarized binary (21 MB); the first `serve` downloads MiniCPM5 2B
(2.7 GB) into `~/Library/Application Support/CoreAIKit/Models`; after that `brew services start`
answers `/health` 0.7–4.6 s later (M4 Max, depending on how much of the model is still in the
file cache). From a checkout the same
server is `swift run -c release systemone serve`,
or `decide-cli serve` in `Examples/Decide`; `decide-cli serve --backend fm` is the same endpoint
over Apple's on-device foundation model (`FoundationModelDecisions`: guided generation, one-hot
probabilities, no calibration — the response says so in `metadata`).
A client written for the hosted endpoint is pointed at this one by its base URL and nothing
else changes. Measured 2026-09-23 against this server: the official TypeSafe Python SDK
(`typesafe-sdk` 0.7.1) with `TYPESAFE_BASE_URL=http://127.0.0.1:8090` and any string as
`TYPESAFE_API_KEY` (or `TypeSafeClient(base_url:)`) got its typed answers back unchanged,
158–201 ms for three questions; the JavaScript SDK takes the same variable or `baseURL`; the
`system-one` SDK on PyPI takes `HTTPConfig(base_url=…)`, 179 ms for three. `GET /v1/models`
answers in the hosted list form (`models`, each `name` / `description` / `release_date`), so
a client's model listing works too. The request and answer forms are in
[`Examples/Decide/README.md`](../Examples/Decide/README.md#the-same-endpoint-your-client-already-speaks).
A choice lists up to the hosted API's 255 options where the model reads that many
([below](#how-many-options-a-choice-lists)).
[`Examples/Decide/conformance/check.py`](../Examples/Decide/conformance/check.py)` <base_url>`
sends 22 requests in those forms and checks every answer's shape, against this server or any
other that speaks the route.

A request can carry one image for a model that reads images, `clef-flash` today: `images` is an
array holding it as a data URL (`"data:image/png;base64,…"`), plain base64, or a path on the
server's machine, which `SystemOneServer` reads only when it listens on 127.0.0.1; `grid` picks the
vision tower's tile, 448 (the default) or 256. Any other model answers such a request with a 422
that names it, and a request without the field reads as before
([every question at once](#every-question-at-once-clef-flash)).

## From a coding agent

`systemone mcp` serves the same decisions as tools of a Model Context Protocol server on
stdin/stdout. Register it once and the agent's own sessions see `decide` and `models`:

```bash
claude mcp add systemone -- "$(brew --prefix)/bin/systemone" mcp     # Claude Code
codex mcp add systemone -- "$(brew --prefix)/bin/systemone" mcp      # Codex
```
```json
{"mcpServers": {"systemone": {"command": "/opt/homebrew/bin/systemone", "args": ["mcp"]}}}
```

(Cursor: that JSON in `~/.cursor/mcp.json`. From a checkout, `.build/release/systemone` after
`swift build -c release --product systemone`, or `decide-cli mcp` in `Examples/Decide`.)
`decide` takes the `/v1/systemone` request above — `state`, `questions`, an optional `model` —
as its arguments and returns the response as `structuredContent` (and as text); `models` lists
the catalog ids that can answer. The model loads on the first call (`--preload` starts it at
launch) and answers one call at a time; when the input closes, the calls in flight are
answered and the process ends, so a one-shot pipe works too (`printf '…' | systemone mcp`).
Asked from a Claude Code session to classify a ticket
with three questions, the tool call round-tripped in 1,064 ms including the model load, 285 ms
of it decisions (M4 Max, 2026-09-23, two model conversions running on the same Mac); from a
Codex session 4,131 ms, 3.7 s of it the load under that load. The server is
`SystemOneMCPServer` in the kit and the message forms `SystemOneMCP`, for an app that is the
MCP server itself.

## What it costs

| | Mac (M4 Max, macOS 27.0) | iPhone 17 Pro (iOS 27.0) |
|---|---:|---:|
| One decision after the state is read (MiniCPM5 2B int8) | 33–65 ms | 63–95 ms |
| Reading a 565-token contract once, then 12 answers | 316 ms + 532 ms | 508 ms + 1,134 ms |
| Twenty tickets × three columns, 60 decisions | 3,963 ms | 7,420 ms |
| A `/v1/systemone` request, 2 questions on a 59-token state, end to end | 204 ms | — |
| One decision, `laya-multilingual` (an encoder, 256-token window, GPU, state shared) | 11.5 ms | 47 ms |
| One decision, `openthai-systemone` (0.8B, its own answer head; a recurrent hybrid on an S = 1 graph, the state checkpointed after its prefix) | 120 ms† | 263 ms‡ |
| One decision, `qwen3.5-2b-decision` (2B, the same S = 1 graph and checkpoint) | 220 ms† | 1,040 ms‡ |

Measured 2026-09-22/24 on the runs in `Examples/Decide/README.md`, which says what each
number is and what else was running; the phone at thermal state "nominal" for the last three rows.
† `decide-cli bench --repeat 6`, eight questions on one state, nothing else on the GPU
(2026-09-24); with every prompt from scratch they take 696 ms and 1,297 ms.
‡ The same bench on the phone with kit 0.7.1 (2026-09-24), at thermal state "nominal":
`openthai-systemone` unplugged, `qwen3.5-2b-decision` on the charger. With every prompt from
scratch they take 1,653 ms and 6,150 ms (`qwen3.5-2b-decision`'s on kit `a88ec6a`, before the
checkpoint: its 0.7.1 from-scratch runs went hot).

## Which model

The default, `minicpm5-2b`, is the fast one: at the median, 0.14 s per question on the Mac and
0.17–0.63 s on the phone, up to 255 options, probabilities read at a fitted temperature. On a
Mac, name `apus-decision-v1-4b` when accuracy matters more than time and download: 0.986 on
JevBench's standard items (the hosted Jev's score), up to 16 options, 5.8 GB, about 2.6 s for
the first question on a short state and 1.2 s for each later question on it. On an iPhone, keep
`minicpm5-2b`: of the five models run there, it scores highest on the hard items (0.459).
Name a model with `options: .model("apus-decision-v1-4b")` in Swift, or
`systemone serve --model apus-decision-v1-4b` for the server. A question about an image goes to
`decider-2b-vision` through `CoreAI.decide(image:…)` ([below](#decisions-about-an-image)), or to
`clef-flash` with the request's `images` ([below](#every-question-at-once-clef-flash)); the tables
here are text only.

JevBench's 231 public items (48 easy, 72 standard, 111 hard), sent one question per request to
`decide-cli serve` by the benchmark's own harness, which also scored them; M4 Max, macOS 27.0,
2026-09-24:

| Model | Standard | Hard | Median per question | Slowest hard item | Download | Most options | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| `apus-decision-v1-4b` | 0.986 | 0.559 | 5.94 s¹ | 217.6 s¹ | 5.8 GB | 16 | Mac only; browser and workflow steps, English + Chinese |
| `system-one-scorer-4b` | 0.861 | 0.505 | 8.96 s¹ | 72.0 s¹ | 5.1 GB | 255 | Mac only; CC BY-NC 4.0; 70 hard states cut to its 384-token rows |
| `decider-0.8b` | 0.833 | 0.414 | 1.69 s | 90.6 s | 1.3 GB | 255 | |
| `openthai-systemone` | 0.819 | 0.324 | 0.62 s | 39.0 s | 1.1 GB | 255 | Thai + English; an abstain probability |
| `qwen3.5-2b-decision` | 0.778 | 0.405 | 3.24 s¹ | 138.5 s¹ | 3.0 GB | 26 | |
| `qwen3.5-2b` | 0.750 | 0.441 | 2.74 s | 62.3 s | 3.0 GB | 26 | chat model, zero-shot, no calibration |
| `minicpm5-2b` (default) | 0.708 | 0.459 | 0.14 s | 1.8 s | 2.7 GB | 255 | chat model, zero-shot, catalog temperature 2.93 |
| `laya-multilingual` | 0.403 | 0.342 | 0.02 s | 0.3 s | 0.7 GB | 20 | encoder; 256-token window: 95 hard states cut |
| `decider-2b-vision` | — | — | — | — | 3.3 GB | 10 | an image and the state, `CoreAI.decide(image:…)`; not behind the server, so not run here ([about an image](#decisions-about-an-image)) |

¹ With two other model servers on the GPU. Re-run alone on 30 items, they gave the same
probabilities in about half the time: 2.6 s per question for `apus-decision-v1-4b` on the easy
and standard ones.
The GPU was shared with other work on every row, so the times are an upper bound. Six of the
eight models are Qwen3.5 recurrent hybrids that read a state one token at a time, so their
slowest hard items take 39–218 s. The hosted Jev (1.13.0) scores 0.986 on the same standard
items and 0.730 on the hard ones, on its own serving stack (JevBench's published per-item
results).

The same requests on an iPhone 17 Pro, answered in an app through `TypedDecisions.systemOne(_:)`,
the path the server takes (iOS 27.0, kit 0.7.1), 2026-09-24; every time here is at thermal
state "nominal". The phone gave the Mac's answer on every item it ran, 882 of 882:

| Model | Standard | Hard | Median per question, standard | Median per question, hard | Download | Most options | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| `decider-0.8b` | 0.833 | 0.414² | 1.33 s³ | 8.04 s² | 1.3 GB | 255 | |
| `openthai-systemone` | 0.819 | 0.324² | 0.99 s | 5.77 s² | 1.1 GB | 255 | Thai + English |
| `qwen3.5-2b-decision` | 0.778 | 0.405² | 3.95 s | 18.09 s² | 3.0 GB | 26 | |
| `minicpm5-2b` (default) | 0.708 | 0.459 | 0.17 s | 0.63 s | 2.7 GB | 255 | chat model |
| `laya-multilingual` | 0.403 | 0.342 | 0.05 s | 0.07 s | 0.7 GB | 20 | encoder; 256-token window |

² The phone ran the three hybrids on the 20 hard items of median length, with the Mac's answer
on each. Their hard accuracy is the Mac's over all 111; their hard time is over those 20.
³ Its easy items. Its standard items ran while the phone was hot, at 2.26 s.
`apus-decision-v1-4b` and `system-one-scorer-4b` have no iPhone variant; `qwen3.5-2b` was not run
on the phone. The Mac's requests and answers are in
[mlboydaisuke/coreai-decision-models-public231](https://huggingface.co/datasets/mlboydaisuke/coreai-decision-models-public231);
[`Examples/Decide/README.md`](../Examples/Decide/README.md#jevbench-public-231-mac-2026-09-24)
has the Mac table with the easy items, ECE and p95.

## What it gets right, and what it does not

`minicpm5-2b` is the default because its int8 decisions track its own full-precision readout:
141/144 argmax agreement on SemIf's authored fixture, mean |Δp| 0.02, family-balanced accuracy
0.681 against 0.686 published. The 4-bit `qwen3-0.6b` does not (59/144) and is documented as
speed-only. `decider-0.8b`, a model trained for these questions, is token-identical to its
author's readout on all 44 of its fixture rows, the 255-option row included; through the hosted
request form (`CoreAI.systemOne`) its 13-request fixture comes back with the author's option on
all 36 answers and every probability, score, level fit and fit mass within 0.02 of the author's
fp32 assembly (max 0.016, on a five-level fit mass).
`openthai-systemone`, a Thai +
English decision model with a 256-way answer head of its own (up to 255 options, an abstain
probability), is token-identical and argmax-identical to its author's readout on all 50 of its
fixture rows (int8 max |Δp| 0.023; on the iPhone 17 Pro the same 50/50 at 0.021) and scores 0.725 on
SemIf's 144 English rows through the kit, at 0.35 s per fixture question on the Mac and 1.5 s per
decision on the phone (1.9 s per fixture question when the phone was hot).
`apus-decision-v1-4b`, a Qwen3.5-4B decision model for browser and workflow steps read at the
letters A–P under its chat template, is token-identical to its author's compiled prompts on the
40 choice and yes/no fixture rows (max |Δp| 0.0055) and scores 0.906 on the same 144 rows, at
about 2 s per decision on the Mac (1.2 s for a later question on a state it has read).
`qwen3.5-2b-decision`, a Qwen3.5-2B decision model with its calibration folded into the weights and
read at space-prefixed letters after a plain-text prompt, is token-identical to its author's rows and
argmax-identical to its fp32 reference on all 58 fixture rows (int8 max |Δp| 0.0079; on the iPhone 17
Pro the same 58/58 at 0.0070) and scores 0.798 on the same 144 rows, at about 1 s per decision on the
Mac (0.2 s for a later question on a state it has read) and about 6 s on the phone (11 s when the
phone is hot), where its S = 1 prefill runs at the phone's rate.
`system-one-scorer-4b`, a Qwen3.5-4B scoring head that reads one row per option at its author's
temperature (CC BY-NC 4.0), is token-identical to its author's 280 fixture rows and argmax-identical
on all 48 questions (int8 max |Δp| 0.011) and scores 0.844 on the same 144 rows, at about 2 s per
three-option decision on the Mac (0.7 s per decision on a state it has read).
`laya-multilingual`, an encoder-type decision model (convaiinnovations' laya, the multilingual
checkpoint: an mmBERT-base encoder with a typed decision head, Apache-2.0), reads the whole question
in one forward pass and answers at a mask marker in front of each option; nothing is generated or
prefilled. Its rows through the kit are token- and marker-identical to its publisher's builder on
all 201 multilingual fixture rows at both of its windows (256 and 512 tokens), and the fp16-weight
bundle is argmax-identical to the official model on all 81 choice and score rows (max |Δp| 5e-6 on
the Mac GPU, 9e-6 on the CPU, at T = 1). On the same 144 rows it answers as the official model does
(144/144 argmax; family-balanced accuracy 0.611 and accuracy 0.590, the model's own figures), at
11.5 ms per decision on the Mac GPU and 47 ms on the iPhone 17 Pro's (53–55 ms per fixture row),
read at the calibration its bundle declares; on the phone the same 201 rows pass (argmax 81/81,
max |Δp| 8e-6). The shipped
bundles run on the GPU: the Neural Engine takes only an fp16-compute graph, which misses the answer
bar, and with a Neural Engine preference the shipped graph misses it too, on the Mac and on the
phone, its answers changing from run to run.

What the probabilities are worth, on the same 144 rows (top-label ECE over 10 equal-width bins,
multi-class Brier): read raw, `minicpm5-2b` is over-confident — ECE 0.167, Brier 0.455 at
accuracy 0.701. So the kit reads it at the temperature its catalog entry records
(`CatalogEntry.calibration`): 2.93, fitted by `decide-cli calibrate` on SemIf's 108 perturbation
rows and reported on the 144 authored rows, where ECE goes to 0.071, Brier to 0.411 and NLL from
0.886 to 0.706. No answer changes; a temperature never moves the argmax. The two sets share no
row but do share situations — every perturbation row was made from one of 36 authored rows — so
the temperature was also checked where nothing is shared: fitted on two of the authored rows'
three task families and read on the third, it comes out 1.69, 2.79 and 2.52, and the 144
held-out rows go from ECE 0.167 to 0.067 (Brier 0.455 → 0.424). One temperature is an average:
per family the ECE goes 0.279 → 0.174 and 0.185 → 0.169, and 0.096 → 0.123 for the family that
was already close. The records were fitted on three-option choices; a yes/no and a score are
read at the same temperature until they are fitted apart. `minicpm5-1b` (9.974) and `qwen3-0.6b`
(11.882) carry records too, and both read nearly flat: the perturbations break the 1B (accuracy
0.435 on them), and the 0.6B answers at chance (0.340 on the authored rows). A model whose
author fitted or folded in a temperature keeps it and has no record — `decider-0.8b` at its
card's 1.03 (accuracy 0.771, balanced 0.753, ECE 0.061, Brier 0.296), the bundles that declare
one, `qwen3.5-2b-decision` at 1. `TypedDecisions.Configuration.temperature` overrides every one
of them; the tables are in [`Examples/Decide/README.md`](../Examples/Decide/README.md#measured).
JevBench's 231 public items on eight catalog models, through `decide-cli serve`: [the table](../Examples/Decide/README.md#jevbench-public-231-mac-2026-09-24).

The shape of the question decides more than the model. On MiniCPM5 2B, measured on the
screens' samples:

- A yes/no on a **short** text leans yes. Ask a choice with a named "none" option, or a
  three-level scale. A yes/no on a long text (a contract) reads correctly.
- "How urgent is this?" on a short row answers "today" for nearly every row, in every shape
  tried. There is no urgency column in the sample table.
- Naming the options alone ("run it / ask / refuse") makes a permission gate ask about 9
  commands of 14; naming each option with the policy's own words gives 6 / 6 / 2, and nothing
  the policy refuses gets through.
- "Which move?" (stay / left / right) drives a car into rocks; "which lane?" with each lane
  described by what lies ahead never picks a rock lane when a clear one is offered. It cannot
  rank two bad lanes, so the road keeps one clear.

## How many options a choice lists

As many as the model reads, up to the hosted API's 255. `TypedDecisions.maxOptions` gives the
count, and the server answers a longer list with a 422 that names it.

| Model | Options are read at | Options |
|---|---|---:|
| `minicpm5-2b` | A–Z; past 26, the numbers 1–255 | 255 |
| `decider-0.8b` | its author's labels A–Z, AA, AB, … | 255 |
| `openthai-systemone` | its 256-way head | 255 |
| `system-one-scorer-4b` | one row per option | 255 |
| `qwen3-0.6b` | A–Z | 26 |
| `qwen3.5-2b-decision` | ` A`–` Z` after its plain-text prompt | 26 |
| `apus-decision-v1-4b` | A–P | 16 |
| `laya-multilingual` | a mask marker before each option, in one forward pass | 20 |
| `decider-2b-vision` | A–J after each question's `Answer: (`, every question of a call in one row ([about an image](#decisions-about-an-image)) | 10 |
| `clef-flash` | a joint head over every option of every question in one pass, 16 questions and 128 options a request ([every question at once](#every-question-at-once-clef-flash)) | 128 |
| `kev-0.8b`, `kev-4b` | a pointer head at each option's closing token, one row per question ([one row per question](#one-row-per-question-kev)) | 255 |

A chat model is read at numbers past 26 because it answers a two-letter label with one of its
letters (`AZ` → `Z`). The numbers need a tokenizer that writes 1–255 as single tokens, as
MiniCPM5's does. Qwen's stops at 9, so a Qwen chat model lists 26. On 61 synthetic rows of
17–255 options, each state naming the answer, `minicpm5-2b` picked the named option on:

| Options | 17–27 | 40 | 52 | 64 | 100 | 128 | 200 | 255 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| numbers past 26 (what the kit does) | 21/21 | 6/6 | 6/6 | 6/6 | 4/5 | 6/6 | 4/5 | 5/6 |
| two-letter labels past 26 | 21/21 | 5/6 | 4/6 | 4/6 | 4/5 | 3/6 | 1/5 | 1/6 |

Measured through the kit on this branch's build (int8, M4 Max, 2026-09-23); the fp32 model picks
the same option on every row. `decider-0.8b` picked it on all 60 rows that fit its 4,096-token
context. A question past 26 options reads its own system line, so it prefills the state again
instead of reusing it.

The iPhone's 1,024-token limit belongs to the pipelined engine that chat generation runs on: on
iOS it caps a growing KV cache there, because the on-device compiler miscompiles that graph once
the cache reaches 2,048 positions. A decision loads a language model on the sequential engine,
which has no such cap. On an iPhone 17 Pro, `minicpm5-2b` answered all 111 JevBench hard items,
39 of them with prompts past 1,024 tokens (up to 3,789), with the Mac's answer on every one
(max |Δp| 0.0046; kit 0.7.1, iOS 27.0, 2026-09-24). A 255-option choice in the chat form,
3,100–4,200 tokens with short options, has not been run on a phone. A decision model takes a
long row there too: `decider-0.8b`'s 255-option fixture row, 1,965 tokens, answered on the
iPhone 17 Pro in 70 s with the fp32 readout's argmax (2026-09-23, all 44 rows argmax-identical
there, max |Δp| 0.009), and `qwen3.5-2b-decision`'s 1,743-token rows in 155 s — on a decision
model a wide choice is a slow call, not a Mac-only one.

## Decisions about an image

`decider-2b-vision` (Mapika, Apache-2.0) is the decider transplanted into Qwen3.5-2B's
vision-language model and fine-tuned on game frames, The Cauldron's multiple-choice image tasks and
a replay of the text mixture. The same three shapes, about an image and a state (`import CoreAIOps`):

```swift
let a = try await CoreAI.decide(
    image: frame, "You control the right paddle. The image shows the current game screen.",
    ["move": .choice("What should you do right now?", ["move paddle up", "move paddle down", "stay"]),
     "ball": .noul("Is the ball moving toward your paddle?")],
    grid: .g256)
a["move"]?.choice          // "move paddle up"
a["move"]?.timing          // imageSeconds (decode, resize, tower), decoderSeconds, and the total
```

Every question of a call is one block of one prompt, and one pass reads every answer at its own
slot: each answer sees the image, the state and all the questions, but no other answer — the
author's multi-question contract (the text models above read one prompt per question). The keys are
laid out in sorted order. A choice lists up to 10 options, the letters A–J; a yes/no is the choice
`no` / `yes`; a score lists its levels as `k: level` and answers Σ k · p(k). `KitVisionDecider` is
the model-level form an app keeps warm, and `decide(image: nil, …)` there reads a text-only row
through the same weights.

The image is resized to a fixed square, its aspect ratio not kept:

| `grid` | tile | image tokens | for |
|---|---|---:|---|
| `.g256` (default) | 256 × 256 | 64 | game frames, speed |
| `.g448` | 448 × 448 | 196 | photos; its 663 MB tower downloads the first time it is asked for |

On the author's own code, a photo read at `g448` agrees with the processor's native grid on 0.980 of
150 Visual7W items and 0.940 of 100 VSR items, against 0.947 and 0.910 at `g256` (the zoo's grid-price
table); on the 256×240 game frames the native grid is `g256`.

The host is the model zoo's `apps/DeciderVision` library, the code its gates ran: the image resized
in Pillow's pass order (a CGContext resize is a different image), the tower baked at the grid, the
author's prompt with the tower's rows as its image block, and a two-function decoder (S = 1 and
S = 16) in the bundle's chunk order, on the low-level runtime. Through `decide-cli parity` on the
zoo's fixture, the model downloaded from the catalog pin (M4 Max, macOS 27.0 26A428, Release,
2026-09-29):

| arm | runs | slots | ids and slots = the author's | argmax = the author's fp32 | full-vocabulary top-1 | max \|Δp\| | mean of run means |
|---|---:|---:|---:|---:|---:|---:|---:|
| `g256` | 35 | 49 | 35/35 | 49/49 | 49/49 | 0.0103 | 0.00027 |
| `g448` | 35 | 49 | 35/35 | 49/49 | 49/49 | 0.0080 | 0.00026 |
| text only | 6 | 10 | 6/6 | 10/10 | 10/10 | 0.0027 | 0.00024 |

Against the zoo's own Swift run of the same files the letter logits are bit-equal on 108/108 slots,
and a second pass repeats them bit for bit. A decision, from the image file to the answers, took a
median 440 ms at `g256` (143 tokens), 797 ms at `g448` (275) and 346 ms text only (76), with another
job holding the GPU at 80–90 % — an upper bound; the zoo measured 322 / 584 / 284 ms on this Mac
with the GPU idle, from its AOT asset. The vision tower is 29 ms (`g256`) and 71 ms (`g448`) of it.

The iPhone has not been run through the kit. The zoo's gate app runs the same library on the same
`.aimodel`s on an iPhone 18 Pro (device JIT, iOS 27.0): 108/108 argmax against the author, and one
game frame in 754–785 ms at `g256` and 1,415–1,439 ms at `g448`, a text row in 725–735 ms. The app
needs `com.apple.developer.kernel.increased-memory-limit` — without it the decoder's first
specialization dies with `std::bad_alloc` — and the runtime's cache for the three graphs takes
6.7 GB on the phone; the decoder loads in 21.6 s the first time and 7.9 s from that cache.

This model is not behind `/v1/systemone`: `systemone serve` and `decide-cli serve` do not load it,
and `systemone models` does not list it. The wire's `images` field reaches `clef-flash`
([below](#every-question-at-once-clef-flash)).

## Every question at once: clef-flash

`clef-flash` (Cloudflare, Apache-2.0, from Qwen3.5-9B) reads a whole request in one pass: the state as text or
JSON and at most one image, then every question with its options. A joint schema head reads the decoder's hidden
state at every position — each question's and each option's span, the last token, and each option's rows of the
model's 2.03 GB lm_head table — and gives each option a logit; each question is a float32 softmax over its own
options at temperature 1. Nothing is generated, and every answer sees every other question. A request takes up to
16 questions and 128 options in all. Mac only: the decoder alone is 15.9 GB. Its catalog kind is `jointDecision`:
`systemone models` lists it as that, `TypedDecisions(catalog:)` refuses it by name before downloading anything, and a
kit released before it (0.7.3 and earlier) decodes the kind as `unknown` and lists it nowhere.

```swift
let decider = try await KitClefDecider(catalog: "clef-flash")          // 18.2 GB the first time
let r = try await decider.systemOne(SystemOne.Request(
    state: "Our checkout started returning errors and orders are blocked.",
    questions: ["department": .choice("Which team should handle the message?", ["billing", "technical"]),
                "outage": .noul("Is a service down?")]))
r["department"]?.choice
```

On the server it is `systemone serve --model clef-flash`, and an image goes in the request's `images` (a data URL,
base64, or a path on the server's machine) with `grid` 448 or 256. The host is the model zoo's `apps/ClefFlash`
library with every numeric path unchanged: the request rendered as the author's `encode_record()`, each piece
tokenized alone; the image resized by Pillow's own integer bicubic (a float resize lands one or two levels off on
some tiles) and read by a tower baked at the grid; the decoder 64 tokens per call from zeroed states; the head graph
and the table. Through `decide-cli parity` on the zoo's fixture, the model downloaded from the catalog pin (M4 Max,
macOS 27.0 26A428, Release, 2026-10-04):

| set | runs | ids and spans = the author's | argmax = the author's fp32 (top-2 margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of run means |
|---|---:|---:|---:|---:|---:|---:|
| fixture (186 records) | 200 | 200/200 | 352/352 | 3/3 | 0.0122 | 0.00057 |
| held out (30 records) | 40 | 40/40 | 186/186 | 2/2 | 0.0082 | 0.00037 |

Against the zoo's own Swift run of the same files, the logits, the probabilities, the decoder's hidden rows and the
responses are bit-equal on all 240 runs, and the first run read again repeats them bit for bit. The fixture's 14
`native` runs, at the processor's own grid, are not run: no tower here has that grid. The zoo's gate fed 13 of them
the author's image rows; `photo_01`'s 1,200 image rows are more than the decoder's 1,024-row buffer.

`systemone serve --model clef-flash` passes `conformance/check.py`, 22 of 22 requests and the three routes (a
choice past 128 options is the 422 that names the count). The fixture's receipt sent in `images`, as a data URL and
as a path, comes back with the zoo Swift run's probabilities to the wire's four decimals, 14 of 14 values. The
answers are in the server's form: `confidence` is 1 − normalised entropy, and `usage` counts the row once per
answer, as for every model here; `metadata.row_tokens` is the row, read once.

## One row per question: Kev

`kev-0.8b` and `kev-4b` (Jared Palmer, Apache-2.0: a rank-16 LoRA adapter and a pointer head on Qwen3.5-0.8B-Base and
Qwen3.5-4B-Base) read each question as its own row: the state, then the question and its options between the author's
delimiter tokens. The author's head compares the decoder's hidden state at the row's last token with the state at each
option's closing token. Each question is a softmax over its own options at the author's temperature, and nothing is
generated. A choice lists up to 255 options, and a row holds at most 3,968 tokens. Kev-0.8B (1.5 GB) runs on the Mac and
the iPhone, Kev-4B (8.4 GB) on the Mac. The catalog kind is `rowDecision`: `TypedDecisions(catalog:)` refuses it by name,
and a kit released before it (0.7.3 and earlier) decodes the kind as `unknown` and lists it nowhere.

```swift
let decider = try await KitKevDecider(catalog: "kev-0.8b")              // 1.5 GB the first time
let r = try await decider.systemOne(try SystemOne.request(from: body))   // a text or JSON state
r["team"]?.choice
let prepared = try await decider.prepare(state: ticket)                  // the state's calls, once
let later = try await decider.decide(prepared: prepared, questionsJSON: questions)
```

On the server it is `systemone serve --model kev-0.8b`; a request with `images` gets a 422 that names the model. The host
is the model zoo's `apps/Kev` library with every numeric path unchanged: the author's rendering of the state and the
options, the decoder 128 tokens per call from zeroed states, and the head on the host in float64. By default a request's
state runs once and each question continues from a copy of the decoder's states; on this graph that gives the same
hidden rows as running each row whole, bit for bit. Through `decide-cli parity` on the zoo's fixture, its 155 records
that carry their text, the model downloaded from the catalog pin (M4 Max, macOS 27.0 26A428, 2026-10-04):

| model | questions | row ids = the author's | argmax = the author's fp32 (top-2 margin > 0.02) | near-ties agreeing | max \|Δp\| | mean of row means |
|---|---:|---:|---:|---:|---:|---:|
| `kev-0.8b` | 186 | 186/186 | 181/181 | 4/5 | 0.0124 | 0.00107 |
| `kev-4b` | 186 | 186/186 | 182/182 | 3/4 | 0.0153 | 0.00081 |

Against the zoo's own Swift run of the same bundle, the hidden rows and the probabilities are bit-equal on every question
(186 of 186 for each model) and the answers on every record (155 of 155); the first record read again repeats them bit
for bit. `systemone serve --model kev-0.8b` passes `conformance/check.py`, 22 of 22 requests and the three routes (a
choice of 255 options makes a row past 3,968 tokens, the 422 that says so). The answers are in the server's form:
`confidence` is 1 − normalised entropy and `usage` counts each question's row; `metadata.packed_tokens` is the author's
count (the state once, then each question).

## What runs

Ten whole uses, one action and the complete result, from the same sources on the Mac and the
iPhone ([`Examples/Decide`](../Examples/Decide)):

| Use | What you do → what you get |
|---|---|
| Autofill | copy an email → every field of a checkout form fills at once |
| Checklist | open a contract → a list of verdicts to your questions |
| Sorter | open a folder → what needs you first, then everything filed; Move files does it |
| Search | a query and passages → ranked by one decision each |
| Speech gate | speak → only the utterances meant for the assistant pass |
| Drive | press Drive → the model drives a car, one lane choice per tick |
| Columns | open a CSV → the columns you asked for, every row; sort, save |
| Command guard | an agent's command log and a policy in plain words → run / ask / refuse (also a PreToolUse hook) |
| Context | an agent transcript's tool results → the unrelated ones dropped, tokens counted |
| Typing | type → tone, intent and an emoji at every pause |

Clips of each on the Mac and on the iPhone are in
[coreai-assets `kit/decide`](https://github.com/john-rocky/coreai-assets/tree/main/kit/decide).

## Where the pieces are

- `Sources/CoreAIKit/Decide/` — `TypedDecisions` (the model-level API: prefill once, decide
  N times; `systemOne(_:)` answers a whole request in the hosted form), `Decision` (the value
  types), `DecisionPrompt` / `DeciderPrompt` / `SlotPrompt`
  (the three renderings: a chat model's JSON turn, a decision model's plain text, a slot-head
  model's control tokens), `SystemOneWire` + `OrderedJSON` (the hosted forms), `SystemOneCall`
  (`SystemOne.Response`: a whole request's typed answers and its wire object), `SystemOneServer` (the
  endpoint over Network.framework — the same code listens in an app, `host: "0.0.0.0"` for the
  local network), `DecisionQueue` (one decision at a time over a shared model), and
  `SystemOneMCPServer` + `SystemOneMCP` (the same decisions as Model Context Protocol tools
  over stdio, and the message forms).
- `Sources/CoreAIKit/Decide/FoundationModelDecisions.swift` — Apple's on-device foundation model
  (FoundationModels, `SystemLanguageModel.default`) as a backend: each question one guided
  generation under the answer's schema, the answer one-hot, no probabilities; `DecisionBackend`
  is what it and `TypedDecisions` share and what `SystemOneServer` takes.
- `Sources/CoreAIKit/Decide/Encoder*.swift` — an encoder-type model (laya): `EncoderPrompt` (the
  publisher's row builder; recent questions' tokens are kept), `EncoderReadout` (its host decoder:
  marker logits, act features, the temperature by question type and option count) and
  `EncoderDecider` (the bundle's `main` and `act` functions; `decideRow` gives the raw numbers).
  `TypedDecisions` is the API over them.
- `Sources/CoreAIKit/DeciderVision/` — decider-2b-vision: `KitVisionDecider` (the model-level API; `readout`
  gives the row, every slot's letter logits and each stage's time), `DeciderVisionPreprocessor` (the image resized in
  Pillow's pass order, cut into the tower's patches), `DeciderVisionPromptRenderer` (the author's prompt and slot
  rule) and `DeciderVisionRuntime` (the tower and the two-function decoder on the low-level runtime), ported from
  the model zoo's `apps/DeciderVision`.
- `Sources/CoreAIKit/ClefFlash/` — clef-flash: `KitClefDecider` (the model-level API and a `DecisionBackend`;
  `readout` gives the row, every option's logit and the hidden state's digest), and the model zoo's
  `apps/ClefFlash` host beside it, ported with every numeric path unchanged (`ClefPromptBuilder`, `ClefRenderer`,
  `ClefImagePreprocess`, `ClefDecoder`, `ClefHead`, `ClefPipeline`).
- `Sources/CoreAIKit/Kev/` — Kev: `KitKevDecider` (the model-level API and a `DecisionBackend`; `prepare(state:)` keeps a
  state for later questions; `readout` gives the rows, each row's hidden digest, and every option's logit and p), and the
  model zoo's `apps/Kev` host beside it, ported with every numeric path unchanged (`KevEncoder`, `KevDecoder`, `KevHead`,
  `KevPipeline`).
- `Sources/CoreAIOps/CoreAI+Decide.swift` — `CoreAI.decide`, the one-call op (`decide(image:…)` for an image);
  `CoreAI+SystemOne.swift` — `CoreAI.systemOne`, the same op in the hosted request and
  response forms (the model from `options`, else the request's `model`, else the default).
- `Sources/systemone` — the `systemone` binary (`serve | ask | models | mcp`), built, signed and
  notarized per tag by `.github/workflows/release.yml`; the Homebrew formula lives in
  [john-rocky/homebrew-tap](https://github.com/john-rocky/homebrew-tap).
- `Examples/Decide/CLI` — `decide-cli ask | bench | oracle | parity | filter | serve | mcp`.
- Android: the same request and answer forms over LiteRT decision encoders are in
  [hfmodels-android](https://github.com/john-rocky/hfmodels-android) (typed-decisions branch).
- Other servers that speak the same route, for a client that switches between them by base
  URL: [jev-rs](https://github.com/yijunyu/jev-rs) (Rust, over `llama-server`),
  [System One Lite](https://github.com/snellingio/system-one) (MLX); the vendor-neutral
  [system-one](https://github.com/asynq-io/system-one) SDK reaches any of them.
