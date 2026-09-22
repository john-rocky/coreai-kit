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
A client library written for the hosted endpoint is pointed at this one by its base URL
(`system-one` on PyPI: `HTTPConfig(base_url="http://127.0.0.1:8090", model="minicpm5-2b")`;
clients that read `TYPESAFE_BASE_URL` or `SYSTEM_ONE_BASE_URL`: set it) and nothing else
changes. The request and answer forms, and the one declared difference (16 options per
choice, not 255), are in [`Examples/Decide/README.md`](../Examples/Decide/README.md#the-same-endpoint-your-client-already-speaks).

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

Measured 2026-09-22/23 on the runs in `Examples/Decide/README.md`, which says what each
number is and what else was running.

## What it gets right, and what it does not

`minicpm5-2b` is the default because its int8 decisions track its own full-precision readout:
141/144 argmax agreement on SemIf's authored fixture, mean |Δp| 0.02, family-balanced accuracy
0.681 against 0.686 published. The 4-bit `qwen3-0.6b` does not (59/144) and is documented as
speed-only. `decider-0.8b`, a model trained for these questions, is token-identical to its
author's readout on the 43 fixture rows the kit can list. `openthai-systemone`, a Thai +
English decision model with a 256-way answer head of its own (up to 255 options, an abstain
probability), is token-identical and argmax-identical to its author's readout on all 50 of its
fixture rows (int8 max |Δp| 0.023) and scores 0.725 on SemIf's 144 English rows through the kit.
`apus-openjev-v1-4b`, a Qwen3.5-4B decision model for browser and workflow steps read at the
letters A–P under its chat template, is token-identical to its author's compiled prompts on the
40 choice and yes/no fixture rows (max |Δp| 0.0055) and scores 0.906 on the same 144 rows, at
about 2 s per decision on the Mac.

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
  N times), `Decision` (the value types), `DecisionPrompt` / `DeciderPrompt` / `SlotPrompt`
  (the three renderings: a chat model's JSON turn, a decision model's plain text, a slot-head
  model's control tokens), `SystemOneWire` + `OrderedJSON` (the hosted forms), `SystemOneServer` (the
  endpoint over Network.framework — the same code listens in an app, `host: "0.0.0.0"` for the
  local network), `DecisionQueue` (one decision at a time over a shared model), and
  `SystemOneMCPServer` + `SystemOneMCP` (the same decisions as Model Context Protocol tools
  over stdio, and the message forms).
- `Sources/CoreAIOps/CoreAI+Decide.swift` — `CoreAI.decide`, the one-call op.
- `Sources/systemone` — the `systemone` binary (`serve | ask | models | mcp`), built, signed and
  notarized per tag by `.github/workflows/release.yml`; the Homebrew formula lives in
  [john-rocky/homebrew-tap](https://github.com/john-rocky/homebrew-tap).
- `Examples/Decide/CLI` — `decide-cli ask | bench | oracle | parity | filter | serve | mcp`.
- Android: the same request and answer forms over LiteRT decision encoders are in
  [hfmodels-android](https://github.com/john-rocky/hfmodels-android) (typed-decisions branch).
