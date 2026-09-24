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
or `decide-cli serve` in `Examples/Decide`.
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
| One decision, `openthai-systemone` (0.8B, its own answer head; a recurrent hybrid on an S = 1 graph, the state checkpointed after its prefix) | 120 ms† | 1,527 ms‡ |
| One decision, `qwen3.5-2b-decision` (2B, the same S = 1 graph and checkpoint) | 220 ms† | 6,063 ms‡ |

Measured 2026-09-22/24 on the runs in `Examples/Decide/README.md`, which says what each
number is and what else was running; the phone at thermal state "nominal" for the last three rows.
† `decide-cli bench --repeat 6`, eight questions on one state, nothing else on the GPU
(2026-09-24); with every prompt from scratch they take 696 ms and 1,297 ms.
‡ Before the checkpoint, when every question re-prefilled its whole row; the phone has not been
measured with it yet.

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

On an iPhone a chat model's prompt must stay under 1,024 tokens (the growing cache of the
pipelined engine the chat models run on there), and a 255-option choice in the chat form does
not: 3,100–4,200 tokens with short options, so on a chat model a choice that wide is a Mac
call; the phone's ceiling there is what fits in 1,024 tokens beside the state, about 60 short
options, estimated from the prompt sizes above (not measured on a phone). A decision model, read
on the logits engine, takes a longer row on the phone: `decider-0.8b`'s 255-option fixture row,
1,965 tokens, answered on the iPhone 17 Pro in 70 s with the fp32 readout's argmax
(2026-09-23, all 44 rows argmax-identical there, max |Δp| 0.009), and `qwen3.5-2b-decision`'s
1,743-token rows in 155 s — on a decision model a wide choice is a slow call, not a Mac-only one.

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
- `Sources/CoreAIKit/Decide/Encoder*.swift` — an encoder-type model (laya): `EncoderPrompt` (the
  publisher's row builder; recent questions' tokens are kept), `EncoderReadout` (its host decoder:
  marker logits, act features, the temperature by question type and option count) and
  `EncoderDecider` (the bundle's `main` and `act` functions; `decideRow` gives the raw numbers).
  `TypedDecisions` is the API over them.
- `Sources/CoreAIOps/CoreAI+Decide.swift` — `CoreAI.decide`, the one-call op;
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
