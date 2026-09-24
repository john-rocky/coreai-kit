# Changelog

All notable changes to CoreAIKit are documented here. The project follows
[Semantic Versioning](https://semver.org) — pre-1.0, minor versions may change API,
patch versions never do. See [`docs/STABILITY.md`](docs/STABILITY.md) for the full
policy.

## [Unreleased]

## [0.7.2] — 2026-09-24

A patch. With the default engine, `ChatSession(model:)` and `init(bundleAt:)` no longer stop the
process on the zoo's decode-only chat models. The examples' lockfiles now pin the swift-huggingface
version the package requires. `exact: "0.7.1"` resolvers move to `0.7.2`; `from:` resolvers pick it
up. Built and gated on macOS 27.0 (26A428) and Xcode 27 (27A266a).

### Fixed

- **`ChatSession(model:)` stopped the process on a zoo decode-only model.** With the default
  `.auto` engine, loading a catalog model by its `ModelID` (as the quickstart does), or a local
  bundle through `init(bundleAt:)`, left prompt chunking on. On the zoo's decode-only ports, whose
  bundle names carry `_decode_` (`qwen3.5-2b`, `lfm2.5-1.2b`, `granite-4.0-h-1b` and the other chat
  models among them), the first prompt of more than one token then ended in a fatal
  shape-substitution error. `ChatSession(catalog:)` was not affected: it takes the catalog's
  `pipelined` hint, which already turned chunking off. Those bundles now prefill one token per step
  on every engine, the rule `TypedDecisions` already applied. `ChatSessionSmokeTests` on the
  `qwen3.5-2b` bundle crashed before the change and passes after it.
- **The examples' lockfiles pinned swift-huggingface 0.9.0**, below the package's own
  `from: "0.11.0"`, so building an example rewrote its `Package.resolved`. They pin 0.11.0, as the
  root does.

## [0.7.1] — 2026-09-24

A patch: a macOS app that links CoreAIKit builds in Release again, and the ChatDemo and Speak
Xcode projects resolve the release they ship in. The `coreai-models` pin moves from 0.2.7-zoo to
0.2.8-zoo: the same arm64 code, plus an x86_64 guard. `exact: "0.7.0"` resolvers move to `0.7.1`;
`from:` resolvers pick it up. Built and gated on macOS 27.0 (26A428) and Xcode 27 (27A266a).

### Fixed

- **A macOS app that links `CoreAIKit` failed its Release build.** A Release build compiles every
  package for all of ARCHS_STANDARD, x86_64 included: Archive does, and so does Run in the
  examples, whose schemes run Release. coreai-models 0.2.7-zoo declares a macOS 26 floor, so
  Xcode compiled it for x86_64 as well. Three of its files use `Float16`, which x86_64 macOS does
  not have: 39 errors in `CoreAILanguageModels`, for the examples that link `CoreAIKit` and for
  any app on 0.7.0. `swift build` compiles for the host's architecture only and passed.
  coreai-models is pinned to `0.2.8-zoo`, which adds upstream's own x86_64 guard to those files.
  The arm64 code is unchanged: `decider-0.8b`'s fixture reads bit-identical, 44 of 44 rows. CI now
  builds `Examples/Transcribe` that way (`example-app-release`).
- **The ChatDemo and Speak apps were built against CoreAIKit 0.4.1.** Since 0.4.2, every release
  moved the examples' `Package.swift` (their CLIs) to the new tag but left `project.yml` and the
  generated Xcode project at `exactVersion: 0.4.1`. The apps opened from a 0.7.0 checkout therefore
  built against 0.4.1. Both projects now resolve 0.7.1, and their READMEs name it.

### Docs

- `Examples/Decide/README.md`: JevBench's 231 public items on eight catalog models (Mac,
  2026-09-24), with a one-line link from `docs/SYSTEM_ONE.md`.

## [0.7.0] — 2026-09-24

Decisions on a recurrent hybrid reuse the state: the catalog's five hybrid decision models
prefill a state once and each later question only its own tokens, 2.5–8.1× faster per decision
on an M4 Max, through coreai-models 0.2.7-zoo's engine checkpoint. `CoreAI.systemOne` is the
hosted System One call as one op; a choice lists up to 255 options where the model can read that
many; `qwen3.5-2b-decision`, `system-one-scorer-4b` and the encoder model `laya-multilingual` join
the catalog, with catalog-fitted temperatures and token-level scoring (`logits(for:)`). A minor:
the 0.6.0 catalog id `apus-openjev-v1-4b` becomes `apus-decision-v1-4b`, a request that repeats a
question id is refused, and the calibrated models read at new probabilities (the same answers);
`exact: "0.6.0"` resolvers move to `0.7.0`. Built and gated on macOS 27.0 (26A428) and Xcode 27
(27A266a); the `coreai-models` runtime pin moves from 0.2.4-zoo to 0.2.7-zoo.

### Removed

- **`openjev-27b`** — withdrawn from the catalog (its Hub repo is now private) pending the source
  model's training-data provenance. It entered the catalog after 0.6.0 (under Added below), so no
  tag carries it.

### Changed

- **`apus-openjev-v1-4b` → `apus-decision-v1-4b`** — the catalog id, display name and Hub repo
  (`mlboydaisuke/APUS-Decision-v1-4B-CoreAI`; the old URL redirects) no longer carry the source
  model's product name. The source attribution stays on the card and in `base_model`.
- **Decisions on a recurrent hybrid reuse the state** — a hybrid (Qwen3.5, LFM2.5, Granite 4) on
  the sequential engine cannot rewind its recurrent state, and every decision used to re-prefill
  its whole prompt; the catalog's `decider-0.8b`, `openthai-systemone`, `qwen3.5-2b-decision`,
  `apus-decision-v1-4b` and `system-one-scorer-4b` are such hybrids. `TypedDecisions` now
  checkpoints the state after a state's prefix — `prefill(_:)`, or the first question on a new
  state — with coreai-models 0.2.7-zoo's `InferenceEngine.checkpoint()`, and each later question
  on the state restores it and prefills only its own tokens. Eight questions on one state on an M4
  Max (2026-09-24), per decision, checkpointed vs every prompt from scratch: 186 vs 825 ms
  (`decider-0.8b`), 120 vs 696 ms (`openthai-systemone`), 220 vs 1,297 ms (`qwen3.5-2b-decision`),
  1,211 vs 3,081 ms (`apus-decision-v1-4b`), 719 vs 5,851 ms (`system-one-scorer-4b`). The answers
  are bit-identical to the from-scratch path on the fixtures of the first three (44, 50 and 58
  rows) and, for the two 4B models, on eight questions over two states with one revisited;
  `minicpm5-2b`, an attention model, reads its rows as before, bit for bit. The first question on
  a new state carries the prefix pass in its timing and reports what that pass reused.
  `logits(for:)` returns to the checkpoint the same way; `prefill(tokens:)` takes none. The
  internal `rewind(to:)` reports the engine's own count after the reset, which on a checkpointed
  hybrid can be the checkpoint's position below the index asked for.
- **coreai-models pinned to `0.2.7-zoo`** (was `0.2.4-zoo`) for the checkpoint above. The bump
  also brings `0.2.5-zoo` — the engine stops at a stop sequence instead of decoding to
  `maxTokens`, so a chat turn on the pipelined engine ends with its answer (a 14-token LFM2.5
  1.2B answer under a 2,048-token cap: 8.3 s → 0.2 s on an M4 Max, the fork's measurement) — and
  `0.2.6-zoo`, which declares a macOS 26 / iOS 26 floor with `@available(macOS 27, iOS 27, *)` on
  everything that touches Core AI; this package's floor stays 27. All lockfiles follow.

### Added

- **`CoreAI.systemOne`** — the hosted System One call as one op: a request in the `/v1/systemone`
  form in (`SystemOne.Request`; the request's bytes with `json:`; or `state:` and `questions:`
  keyed by the caller's ids, in the order the answers come back) and the response in that form
  out. `SystemOne.Response` holds the typed answers in request order (`response["queue"]?.choice`),
  `usage`, `stateTokens`, the whole request's `milliseconds`, and the wire object as `value` /
  `dumps()` — byte for byte what `systemone serve` returns. The model is `options.model`, else the
  request's own `model`, else `CoreAI.decide`'s default, from the same cache under the same
  one-request-at-a-time rule; bytes the wire refuses throw `SystemOne.WireError` before any model
  loads. `TypedDecisions.systemOne(_:)` is the model-level call: every question checked against the
  model's option count first, the state prefilled once, each question decided against it.
  `SystemOne.Request` gains initialisers for a request built in Swift, a structured state included
  (serialized the reference way). `SystemOneServer`, `SystemOneMCPServer` and `systemone ask` now
  answer through this one path; their replies are unchanged. Checked against the decider's own API
  on the zoo's 13-request fixture (`SystemOneDeciderSmokeTests`, opt-in with `KIT_DECIDER_FIXTURE`):
  on `decider-0.8b` every one of the 36 answers names the author's option, and every probability,
  expected score, level fit and fit mass is within 0.02 of the author's fp32 assembly (max 0.016, on
  a five-level fit mass; M4 Max, 2026-09-23, GPU shared).
- **A choice lists up to 255 options**, the hosted API's width, where the model can read that
  many. `minicpm5-2b` reads A–Z and, past 26 options, the numbers "1"…"255" under a system line
  that asks for the number; `decider-0.8b` reads its author's labels A–Z, AA, AB, … (its 44-row
  fixture now passes whole, the 255-option row included); `system-one-scorer-4b` scores 255 rows
  (was 64). `TypedDecisions.maxOptions` is each model's own count: 26 on `qwen3-0.6b`, whose
  tokenizer has no single-token numbers past 9, and 16 on `apus-decision-v1-4b`. A question of
  16 options or fewer renders token for token as before. Why numbers: on 61 synthetic rows of
  17–255 options, `minicpm5-2b` picked the named option on 58 read at numbers and on 43 read at
  two-letter labels, where it answers one letter of the label. A 255-option prompt is 1,965
  tokens on the decider's fixture row and 3,100–4,200 in the chat form, over the 1,024 a chat
  model's prompt may take on an iPhone, so on a chat model it is a Mac call; the decider's row
  ran on the iPhone 17 Pro in 70 s (2026-09-23).
- `decide-cli oracle --dump-prompts <path>` writes each row's token ids and answer-slot ids in
  the `--prompts` shape. A row whose prompt outgrows the model's context is skipped and listed
  instead of ending the run.
- **Encoder-type decision models: `Decision.Format.encoder`.** A bundle whose `metadata.json`
  declares `decision.head == "encoder"` (laya multilingual) is recognised by both `TypedDecisions`
  initialisers before anything reads it as a language bundle, and answers in one forward pass per
  question, read at the mask marker in front of each option, at the temperature the bundle declares
  per question type and option count. `prefill` tokenizes the state once; recent questions' tokens
  are reused. `TypedDecisions.Configuration.computeUnits` picks the graphs' compute units (the GPU
  by default). `EncoderPrompt`, `EncoderReadout` and `EncoderDecider` are the low level
  (`decideRow` for the raw numbers). `decide-cli parity` reads `coreai-encoder-fixtures/1` — tokens
  and markers with only a tokenizer (`--tokens-only --tokenizer`), the raw logits with a bundle —
  and `--compute` reaches every command. On the iPhone 17 Pro the catalog's `ios/wfp16-s256`
  bundle passes the same 201 rows (argmax 81/81, max |Δp| 8e-6) at 47 ms per decision on the GPU
  with the state shared (53–55 ms per fixture row);
  a Neural Engine preference misses the bar there too (`Examples/Decide/README.md`).
- `Examples/Decide/conformance/` — `check.py <base_url>`: 22 requests in the hosted System One
  forms and the shape each answer must come back in (types and keys), for this server or any
  other that speaks the route; `calibration.py`: accuracy, NLL, Brier, top-label ECE and a
  fitted temperature from `decide-cli oracle` output on SemIf's `authored144`.
- `CatalogEntry.calibration` (`CatalogEntry.Calibration`: `temperature`, `byType`,
  `temperature(forType:)`): the temperature a model's decisions are read at by default, fitted
  by the maintainer. catalog.json keeps what it was fitted and reported on (`fit`, `report`) in
  the same object; the kit reads only the temperatures.
- `DecisionCalibration`: `rescale` (a distribution re-read at another temperature — exact when it
  is one softmax), `fitTemperature` (least NLL on `calibration.py`'s grid), `metrics` and
  `familyMetrics` (accuracy, balanced accuracy over the right options' ids, NLL, multi-class
  Brier, top-label ECE), each defined as the script defines it.
- `decide-cli calibrate --fit <rows> --report <rows>`: a temperature per question type fitted on
  one labelled fixture and read on another, before and after, per family. It refuses a row id in
  both sets and prints what else they share (situations, source rows, states); `--record`
  writes the catalog record, `--out` / `--out-raw` the rows `calibration.py` reads.
- **`decide-cli --no-share`, `--engine-log` and `parity --dump-probs`** — `--no-share` prefills every
  prompt whole (`TypedDecisions.Configuration.sharePrefix = false`) for any command, the
  from-scratch side of a comparison; `--engine-log` prints the inference engine's own log lines (a
  checkpoint's size and copy time among them); `parity --dump-probs <rows.jsonl>` writes each
  row's probabilities, argmax, prompt and reused tokens, and time.
- **`qwen3.5-2b-decision`** — chaoliangUNSW's Jev-Style-Qwen3.5-2B-Decision (Qwen3.5-2B-Base
  fine-tune, English, Apache-2.0; its calibration temperature folded into the weights, the author
  reports ECE 0.017) as a catalog `decision` model with its readout: `Decision.Format.decisionFunction`,
  the author's plain-text prompt without a chat template read at the space-prefixed letters ` A`–` Z`
  (up to 26 options), T = 1. On the author's 58-row fixture the kit's rows are token- and
  slot-identical to the author's and argmax-identical to the fp32 reference on 58/58 for both
  bundles (int8 max |Δp| 0.0079, fp16 0.0058; `decide-cli parity`); 0.798 mean family balanced
  accuracy on SemIf's 144 English rows through the kit. Mac and iPhone (2.9 GB int8; no iPhone
  number yet).
- **`system-one-scorer-4b`** — pngwn's system-one-qwen3.5-4b-scorer (a Qwen3.5-4B-Base LoRA with
  a scalar scoring head, English, **CC BY-NC 4.0**) as a catalog `decision` model with its readout:
  `Decision.Format.scalar`, one `State:` / `Question:` / `Option:` row per option with the state cut
  from its end to fit 384 tokens, the rows' scalars softmaxed at the author's temperature 1.75 —
  both declared by the bundle's metadata (`decision.head == "scalar"`), which is how the format is
  resolved. Up to 64 options; a yes/no is the rows `yes` / `no`, a score one row per level. On the
  author's 48-question, 280-row fixture the kit's rows are token- and slot-identical on 280/280 and
  argmax-identical to the fp32 readout on 48/48 for both bundles (int8 max |Δp| 0.0110, fp16 0.0037;
  `decide-cli parity`, which reads `coreai-scalar-fixtures/1`); 0.844 mean family balanced accuracy
  on SemIf's 144 English rows through the kit. Mac only (5.1 GB).
- **`Decision.Format.letterList`** — the lettered option list under the chat template that OpenJev's
  helper sends (`State:`, `Question:`, `Options:` as `[A] key: description` lines, "Answer with the
  letter of the best option only."), read at the bare letters A–Z then a–z (up to 52) at the
  temperature the bundle declares, a yes/no calibrated the helper's way (`decision.readout ==
  "letters"` with `temperature` and `noul` in metadata.json); `decide-cli parity` reads the helper's
  fixture rows.
- **`openjev-27b`** — OpenJev (the OpenJev project's Qwen3.8-27B fine-tune; English, German, French,
  Hindi, Chinese, Japanese; **CC BY-NC 4.0** for the weights), the open model behind its
  `/v1/systemone` helper, as a catalog `decision` model with `format: letterList`. On the author's
  61-row fixture the kit's rows are token-, slot- and argmax-identical to the helper's own readout
  on 61/61 (int8 max |Δp| 0.0003; a yes/no compared after the helper's calibration); 0.907 mean
  family balanced accuracy on SemIf's 144 English rows through the kit. Mac only (28 GB int8hu,
  about 28 GB resident, about 8 s per decision on an M4 Max).
- **`CatalogEntry.license`** — the SPDX id of a model's weights license when it restricts use
  (`CC-BY-NC-4.0`); nil for the permissive ones. `systemone models` prints it beside the name and
  `/v1/models` adds it to the description; the zoo card has every model's exact terms.
- **Token-level scoring on `TypedDecisions`** — `logits(for:)` feeds a token sequence and
  returns the logits at its last position for the whole vocabulary (`Decision.Logits`), with
  the KV-cache prefix reuse `decide` has (`timing.reusedTokens`); `prefill(tokens:)` is the
  token-level `prefill(_:)`; `tokenizer` is the bundle's own, to render the prompt with. For
  a caller with its own readout — AnyDecisionModel's Core AI backend sums the variants of
  each label, measures the allowed-answer mass and calibrates, where `decide` takes a softmax
  over one letter token per option. `decide` is unchanged.

### Changed

- A `/v1/systemone` request that keys two questions by the same id is refused — 422,
  `duplicate question id 'q'` — instead of answering both under one key, where the second
  overwrote the first in the response object.
- `SystemOne.maxOptions` is the hosted API's 255 (was 16). `SystemOne.request(from:maxOptions:)`
  takes the loaded model's count, and `SystemOneServer` passes it, so a list the model cannot
  read is a 422 that names the count ("this engine lists at most 26 options, got 27"). The wire
  never takes more than 255. The MCP `decide` tool describes 2–255 options.
- `TypedDecisions` — and so `CoreAI.decide`, `systemone` and `SystemOneServer` — reads a catalog
  model at its entry's `calibration` when it has one. The order is `Configuration.temperature`,
  the catalog, the bundle's declaration, the prompt form's default. `minicpm5-2b` (2.93),
  `minicpm5-1b` (9.974) and `qwen3-0.6b` (11.882) carry records fitted on SemIf's
  `perturbations108`: their probabilities change, their answers do not. `minicpm5-2b` on
  `authored144`: ECE 0.167 → 0.071, Brier 0.455 → 0.411, accuracy 0.701 either way. A local
  bundle (`init(bundleAt:)`) has no catalog entry and reads as before.

### Fixed

- **`decider-0.8b` read a described option twice over the wire.** `SystemOne.request(from:)`
  writes each option's description as `key: description`, the text the chat form reads, and the
  decider form composed it again: the model read `repair: repair: Replace damaged parts`, and
  P(repair) on the fixture's one described choice came out 0.941 against the author's 0.971
  (0.972 read in Python from the same bundle). The decider form now keeps a description that
  already carries its name, as the slot, scalar and letter-list forms do; a choice built in Swift
  from separate fields renders as before. This reached every door the wire feeds — `systemone
  serve`, `systemone mcp`, the `decide` MCP tool — for a `choice` whose criteria carry
  descriptions; the chosen option did not change on the fixture.
- **Concurrent calls on one `TypedDecisions` could drive its engine at once.** The actor did
  not serialize a call: each engine `await` let another `decide` or `logits(for:)` call in to
  rewind or feed the engine under the first, which then crashed in `CoreAISequentialEngine` or
  came back without logits. Engine calls now wait their turn in a FIFO async lock, held from
  the rewind to the end of the stream and around `reset()`; a task cancelled while it waits
  throws `CancellationError` and leaves the queue (#43, @mattt).

### Docs

- `docs/SYSTEM_ONE.md`, `Examples/Decide/README.md`, README: the official TypeSafe SDKs
  measured against the local server by base URL alone; other servers that speak the form;
  the calibration table for `minicpm5-2b` and `decider-0.8b`.
- `docs/SYSTEM_ONE.md` "How many options a choice lists": the count per model, the measured
  wide-choice table for `minicpm5-2b`, and the iPhone rule (a chat model's prompt under 1,024
  tokens; a decision model on the logits engine reads longer rows there — measured 2026-09-23).
- `Examples/Decide/README.md`, `docs/SYSTEM_ONE.md`: the iPhone 17 Pro measured (2026-09-23) for
  `laya-multilingual` (201/201 rows, 47 ms per decision on the GPU; a Neural Engine preference
  misses the bar there too), `qwen3.5-2b-decision` (58/58 rows, max |Δp| 0.0070; 6 s per
  decision cool, 11 s hot — the S = 1 prefill at the phone's rate), `openthai-systemone` (50/50 rows, 0.021;
  the same 109/144 on SemIf's rows as the Mac), `decider-0.8b` (44/44 rows with the 255-option
  row) and `minicpm5-2b` on SemIf's rows (143 of 144 answers the Mac's, the one a tie).

## [0.6.0] — 2026-09-23

The System One server as one signed, notarized binary: `brew install john-rocky/tap/systemone
&& systemone serve`, and `brew services start systemone` to keep it running; `systemone mcp` serves the same decisions
to a coding agent over the Model Context Protocol. `SystemOneServer`
and `DecisionQueue` become public API, and `GET /v1/models` answers in the hosted list form
(the official TypeSafe SDK's `models.list()` reads it). A minor: API added, one example file
removed; `exact: "0.5.0"` resolvers move to `0.6.0`. Built and gated on macOS 27.0 (26A428)
and Xcode 27 (27A266a); the `coreai-models` runtime pin stays 0.2.4-zoo.

### Added

- **`systemone` — the System One server without a Swift toolchain.** A root executable
  product (`swift build -c release --product systemone`): `systemone serve [--model] [--host]
  [--port]` is the `/v1/systemone` endpoint, `ask` one decision from the shell (`--json`
  prints the wire form; the state comes from `--state`, `--state-file` or stdin), `models`
  what can decide and what is downloaded. `.github/workflows/release.yml` builds it per tag on
  the self-hosted Mac, signs it with Developer ID, notarizes it and attaches
  `systemone-<tag>-macos-arm64.zip` to the tag's GitHub Release; the formula in
  [john-rocky/homebrew-tap](https://github.com/john-rocky/homebrew-tap) installs that zip
  (`brew install john-rocky/tap/systemone`) and `brew services start systemone` keeps it on
  `127.0.0.1:8090` under launchd.
- **`systemone mcp`** — the typed decisions as tools of a Model Context Protocol server on
  stdio, so a coding agent calls the model on this machine: `claude mcp add systemone --
  "$(brew --prefix)/bin/systemone" mcp` (Codex: `codex mcp add …`; Cursor: `~/.cursor/mcp.json`;
  `decide-cli mcp` from a checkout). Tools `decide` (the `/v1/systemone` request form as
  arguments, the response as `structuredContent` and as text) and `models` (the catalog ids
  that can answer). Both revisions of the protocol are served — the `initialize` handshake of
  2025-11-25 and earlier (what Claude Code 2.1 and Codex 0.154 send) and the per-request
  `_meta` form of 2026-07-28 (`server/discover`). `SystemOneMCPServer` (the server over any
  input/output pair; `run()` returns when the input closes, after the calls in flight have
  answered, so a one-shot pipe works) and `SystemOneMCP` (the message
  forms) are public API in `CoreAIKit`, with hermetic tests over pipes. On a Mac with two model
  conversions running alongside, a three-question classification asked from a Claude Code
  session round-tripped in 1,064 ms including the model load, 285 ms of it decisions.
  `SystemOne.request(from: JSONValue)` joins the bytes overload.
- **`SystemOneServer` and `DecisionQueue` are public** (`CoreAIKit`). The HTTP/1.1 server over
  Network.framework that `decide-cli serve` carried moves into the kit, so an app serves the
  same endpoint itself (on an iPhone, `host: "0.0.0.0"` offers it to the local network);
  `run()` now returns when `stop()` is called and throws when the listener fails instead of
  exiting the process. `DecisionQueue` is the one-at-a-time funnel any concurrent caller of
  one `TypedDecisions` needs. `TypedDecisions.supports(_:)` says whether a catalog entry can
  decide here.
- **`openthai-systemone`** — iApp's OpenThai-SystemOne (Thai + English, Qwen3.5-0.8B tower,
  Apache-2.0) as a catalog `decision` model, and the readout it needs: `Decision.Format.slot`.
  A slot-head model replaces the LM head with a 256-way head read at a `<|ts_answer|>` control
  token — option i is slot i, the last slot means "none of these" — so a choice may list up to
  255 options (`TypedDecisions.maxOptions`; the letter readouts keep 16) and every answer
  carries `Decision.Answer.abstain` (also written to a `/v1/systemone` choice answer). The
  bundle declares the head and a temperature per question type in its `metadata.json`
  (`decision` block); `TypedDecisions.temperature(for:)` reports which applies. On the
  author's 50-row fixture (Thai, English and JSON states; 2–16, 40 and 255 options) the kit's
  rows are token-, slot- and argmax-identical to the author's fp32 readout on 50/50, int8 max
  |Δp| 0.0226 (`decide-cli parity`); 0.725 mean family balanced accuracy on SemIf's 144
  English rows through the kit.
- **`apus-openjev-v1-4b`** — APUS AI Lab's APUS-OpenJev-v1-4B (Qwen3.5-4B, English + Chinese,
  Apache-2.0), a decision model for browser actions and workflow steps, as a catalog `decision`
  model with its readout: `Decision.Format.sharedState`, one `Shared state:` + JSON task user
  turn under the chat template read at the letters A–P, no calibration (the author's own
  statement). A catalog entry may now name a decision model's readout (`CatalogEntry.format`),
  read after the bundle's own declaration and before the kind's default. On the author's
  48-row fixture the kit's prompts are token-identical to the author's compiled ones on all 40
  choice and yes/no rows, argmax 40/40, max |Δp| 0.0055 (`decide-cli parity`, which also reads
  the letter fixture form, `coreai-letter-fixtures/1`); 0.906 mean family balanced accuracy on
  SemIf's 144 English rows through the kit. Mac only (5.8 GB).
- `decide-cli --bundle <dir>` — any command on an unpublished bundle directory; `parity` reads
  the slot fixture form (`coreai-slot-fixtures/1`) and renders JSON states itself.

### Changed

- `Examples/Decide/CLI/Serve.swift` is gone; `decide-cli serve` is a shell over the kit's
  `SystemOneServer` with the same flags and routes.

### Fixed

- Thai, and every script that writes vowels and tones as combining marks, tokenizes through
  swift-transformers differently from the reference tokenizer: the `Split` pre-tokenizer regex
  is applied through Foundation's string search, whose matches snap to grapheme clusters, so a
  letter run keeps its marks and BPE merges differently (23 of the 50 OpenThai fixture rows).
  The slot-head path cuts each text segment with the same regex through ICU on UTF-16 and
  encodes the pieces one by one (`SlotPrompt.Encoder`). The chat and decider paths still
  tokenize through swift-transformers as before.

## [0.5.0] — 2026-09-23

Typed decisions — System One on device — enter the release train: `CoreAI.decide` /
`TypedDecisions`, the `/v1/systemone` forms and `decide-cli serve`, and the ten whole uses
in `Examples/Decide`. A minor: API added, nothing removed; `exact: "0.4.2"` resolvers move to
`0.5.0`. Built and gated on macOS 27.0 (26A428) and Xcode 27 (27A266a); the `coreai-models`
runtime pin stays 0.2.4-zoo.

### Added

- **Typed decisions** — `CoreAI.decide(state, questions)` (CoreAIOps) and `TypedDecisions`
  (CoreAIKit): a state and typed questions (`choice` of 2–16 options, `score` on 2–10 ordered
  levels, `noul` = yes/no) in, answers with probabilities out; nothing is generated, each
  question is one prompt scored at its answer slot. `TypedDecisions.prefill(_:)` runs the
  shared state prefix once and every question rewinds to it (`Decision.Timing.reusedTokens`
  reports what was kept; recurrent hybrids fall back to a full re-prefill). Loads on the
  sequential engine (or static-shape for a Neural Engine bundle) — the engines that return
  logits. Default model `minicpm5-2b`: on SemIf's 144 authored rows its int8 readout agrees
  with the published bf16 argmax on 141/144 (mean |Δp| 0.020, family-balanced accuracy 0.681
  vs 0.686 published); the official 4-bit `qwen3-0.6b` does not (59/144) and is documented
  as speed-only. `Examples/Decide`: a speech gate, a clipboard check with two Shortcuts
  actions, and a passage reranker, each decision with its measured milliseconds, plus
  `decide-cli` with `bench` and `oracle`.
- **A model trained for decisions** — `decider-0.8b` (catalog kind `decision`, the first of
  that kind: a Qwen3.5-0.8B-Base fine-tune that answers typed questions at an answer slot and
  cannot chat). `TypedDecisions` renders it in the form it was trained on
  (`Decision.Format.decider`: `Context:` / `Question:` / `Options:` / `Answer: (`, no chat
  template; a score question is one yes/no row per level, `Decision.Score.fit` keeps the
  per-level P(fits)) at its card's temperature 1.03; chat models keep the JSON-turn form
  (`.chat`). The format follows the catalog kind, `Configuration.format` overrides it for a
  local bundle. On the model's own 44-row fixture the kit's token ids, answer slots and
  probabilities are checked by `decide-cli parity`: on the 43 rows the kit can list (the
  255-option row is beyond its 16), token ids, answer slots and argmax all 43/43, max |Δp|
  0.0088 / mean 0.0009 against the author's fp32 readout (int8 bundle, M4 Max, 2026-09-22).
- `Examples/Decide` becomes five whole uses that run the same `TypedDecisions` on a Mac as on
  an iPhone: **Autofill** (copy an email and every field of a checkout form fills at once —
  the text prefilled once, one choice-among-lines question per field), **Checklist** (a
  contract read once, a list of verdicts to `noul:` / `choice:` / `score:` lines), **Sorter**
  (a folder read once per file: what needs you, then everything filed; Move files does it),
  plus the passage reranker and the speech gate. The clipboard screen folds into Autofill;
  the two Shortcuts actions stay. `decide-cli` gains `filter` (one decision per stdin line — a
  semantic grep) and `parity`; the app takes `-autoplay <screen>` (`-trigger`, `-log`, `-feed`)
  for hands-off runs and recordings.
- Five more whole uses in `Examples/Decide`, the shapes of the most-viewed System One posts
  with the model on the device: **Drive** (the model drives a car down a three-lane road, one
  choice per tick, 80 ticks in 25 s at a 40 ms median on an M4 Max), **Columns** (a CSV and
  the columns you ask for — every row read once, one decision per column; sort, save),
  **Command guard** (an agent's command log under a policy in plain words → run / ask /
  refuse; `hooks/claude-code-guard.sh` runs the same decision as a Claude Code PreToolUse
  hook), **Context** (an agent transcript's tool results scored against the question — the
  unrelated ones drop out, tokens counted by the model's tokenizer) and **Typing** (tone,
  intent and an emoji read from the text at every pause). The README records which
  question shapes read correctly on MiniCPM5 2B and which did not.
- **The `/v1/systemone` forms** — `SystemOne.request(from:)` / `SystemOne.response(model:answers:)`
  (CoreAIKit) read and write the request and answer JSON of the hosted System One API
  (`state` as a string or structured data, `questions` keyed by id with `type` /
  `instructions` / `criteria`; `answers` with `choice` + probabilities, `score` + legend,
  `noul`, `confidence`, `usage`), over an order-keeping `JSONValue` that writes what Python's
  `json.dumps(…, ensure_ascii=False)` writes, so a structured state reaches the model as the
  same bytes a Python client sends. `decide-cli serve` puts a loaded model behind that
  endpoint on this machine (`GET /v1/models`, `GET /health`, CORS open, `--host 0.0.0.0` for
  the local network); a client written for the hosted endpoint switches by base URL. One
  declared difference: at most 16 options per choice. `Examples/Decide/clients/` holds a curl
  and a Python request.

### Changed

- `TypedDecisions.Configuration.temperature` is optional: `nil` (the default) is the model's
  own — 1 for a chat model, the card's calibration for a decision model.
- `TypedDecisions.promptTokens(_:_:)` is `promptRows(_:_:)` and returns one token sequence per
  scored row (a score question under `.decider` is several).

## [0.4.2] — 2026-09-15

Built and gated on the release macOS 27 (26A428) and release Xcode 27 (27A266a); no API
changes, so `from: "0.4.1"` resolvers pick it up automatically. Device numbers in this
train were measured on the iOS 27 RC (24A435); the iOS 27 release build is 24A437 and has
not been re-measured yet.

### Fixed

- Gemma 4 E4B, 12B and 31B catalog pins move to re-saved decode bundles (`gemma-4-E4B-CoreAI` e9ba305a,
  `Gemma-4-12B-CoreAI` 266c0458, `Gemma-4-31B-CoreAI` 92bcdec5). The previously pinned bundles carried coreai-torch 0.4.0-era IR,
  which every OS 27 build since beta 2 refuses at `AIModel.load` (`LLVM ERROR: cannot unwrap empty
  odiec_module_t`), re-measured on the macOS 27 RC 26A428 on 2026-09-14. The re-saved bundles are
  the same weights with debug locations stripped and a `coreai-core 1.0.0b2` producer stamp
  (coreai-model-zoo `conversion/recovery/`).

## [0.4.1] — 2026-09-09

A patch release of the existing public API, validated on macOS 27 beta with Xcode 27
beta 5. This is not a claim of GA validation. Start from the
[versioned quickstart](https://github.com/john-rocky/coreai-kit/tree/0.4.1#quickstart).

- The starter is the catalog's Qwen3 0.6B pin, with measured download sizes and an
  explicit OS/SDK requirement. ChatDemo and Speak now depend on the public exact
  0.4.1 package in both SwiftPM and Xcode; their local-development lockfiles are removed
  so a new clone resolves the release rather than a sibling checkout.
- The runtime and model pins already on main are retained. The fixes below, previously
  available only on main, are now distributed together.

### Added

- `ModelStore(hubBaseURL:)` and `ModelStore(directory:hubBaseURL:)` select an
  HF-compatible endpoint for both model listing and file downloads ([#1](https://github.com/john-rocky/coreai-kit/issues/1)).
  The default Hub, pinned revisions and cache layout are unchanged. HF credentials are
  not copied to the custom host; URL-embedded credentials, query and fragment are rejected.

### Changed

- **The model-fork pin moves to `0.2.4-zoo`** (was `0.2.2-zoo`) — in the manifest and in every
  lockfile: the 27 SwiftPM `Package.resolved` files and the SiriAsk Xcode workspace lockfile,
  which the 0.4.0 sweep did not reach and which still recorded `0.1.1-zoo`. `0.2.3-zoo` is
  upstream `main` through apple/coreai-models#207 (2026-08-28) with the zoo patches rebased on
  top; the tags before it sat on upstream at #90 (just after 0.2.0). The change a running app can see is
  upstream #121: the pipelined composite sampler reused one execution descriptor across
  in-flight steps and garbled output at temperature > 0 (word repetitions, doubled
  punctuation). No tag before `0.2.3-zoo` has that fix. `0.2.4-zoo` then drops the fork's own
  sampler-ordering drain, which that upstream fix made redundant — on a Mac, qwen3-0.6b decodes
  **+43–52% faster** (drain vs. none, interleaved, greedy output token-identical, 12-turn
  reset/generate loops clean); iOS is not measured yet. One call site followed upstream #122's
  rename (`CoreAIRunner(from:)` → `CoreAIRunner(bundle:)`); nothing in the kit's own API moved.

### Fixed

- **Hub tree listings retry transient HTTP 429/5xx failures** up to five attempts,
  with cancellable 2/4/8/16-second waits. Exhaustion reports the final HTTP status;
  only 404 reports a missing variant, while other permanent errors fail immediately.
  File-download retries are unchanged. Adapted from
  [AurionRodgerDiablo's PR #2](https://github.com/john-rocky/coreai-kit/pull/2).

- **Long CJK streaming no longer starves.** Every streaming decode loop held text back
  while the decode contained U+FFFD *anywhere*; one stray raw-byte token — deterministic
  in long Japanese output on qwen3 — made that hold permanent, so no event dispatched for
  the rest of the turn and a FoundationModels session failed with *Session ended without
  producing a response*. The copy-pasted loops are now one `StreamingDetokenizer` that
  holds only a trailing in-progress character and emits an interior replacement character
  as-is — the same contract as the fork's `VanillaDecodingStrategy`, which is why
  `ChatSession` never starved. Reproduced and re-verified on a long English→Japanese
  translation on qwen3-0.6b through `LanguageModelSession` (before: 0 stream snapshots;
  after: the full translation streams). Applies to the FM text/vision/audio/Gemma
  executors, the ASR executor, and the guided loop.

- **Hybrid models kept their second turn.** `0.2.3-zoo` refuses a partial `reset(to:)` on
  models with recurrent state (Qwen3.5/3.6, LFM2.5, Granite 4, Nemotron-H — upstream #132
  throws `invalidState` where the earlier fork tags quietly fell back to a full reset), and
  the kit asks for exactly that rewind on every turn after the first. Reproduced on
  Qwen3.5-0.8B: turn 1 answered, turn 2 failed with *Partial reset is not supported for
  hybrid models with recurrent state*. `ChatSession` and the FoundationModels executor now
  rewind through one helper that falls back to a full reset when the engine says it cannot
  rewind, and report the tokens actually kept. A full reset costs a re-prefill of the
  prompt, nothing else — the kit feeds the full sequence either way.

## [0.4.0] — 2026-08-25

### Added

- **`CoreAI.tidyTranscript(_:)` + `KitTextNormalizer` — the other half of the dictation
  path.** The kit's ASR models all produce raw transcripts; nothing turned one into text a
  person would send. S1-mini by Superwhisper (catalog `s1-mini`, 796 MB, iPhone-verified)
  drops fillers, resolves false starts to whatever the speaker landed on, punctuates, and
  writes spoken numbers, dates, times, currency and email addresses out. Steered by the
  model's own three trained axes — `styling:` / `structure:` / `context:` — because it has no
  free-text instruction.

  It is deliberately **not** a flag on `proofread`, which is contracted to keep the wording as
  close to the original as possible: this op deletes, is English-only, and returns the empty
  string for filler-only input. And it does not ride `ChatSession`: that path sends an empty
  system prompt so every text op can share one model, while S1-mini's system prompt and
  control line are its trained input format, and `enable_thinking=False` is mandatory (leave
  thinking on and every call returns "" — a pipeline that looks like it works).

  **Long input is chunked and stitched**, which is the substance rather than a nicety: on iOS
  the shipped engine caps a growing KV cache at 1024 tokens, and the rewrite runs about as
  long as its input, so a whole meeting transcript passed in one call stops mid-sentence.
  Measured on iPhone 17 Pro — a 611-token transcript produced 413 tokens, every one
  token-identical to the Mac, then stopped at absolute position exactly 1024. Input is cut at
  word boundaries into ~450-token pieces, evenly rather than packed. `Examples/Tidy` is the
  runner (GUI + `tidy-cli`).

- **`CoreAI.capability(_:)` — the question the kit could not be asked.** `prepare` is an
  instruction ("fetch this now"); before an app can decide whether to *offer* a feature it
  needs "can this happen, and what will it cost the user". Answers `.ready`,
  `.needsDownload(bytes:)`, `.needsSystemAssets(what:)`, `.insufficientStorage(needs:free:)`
  or `.unsupportedDevice(reason:)` — offline, without touching the network, so a gallery of
  every op is not 23 round trips. `capabilities()` answers for all of them at once.

  `.needsSystemAssets` is new against the design: with transcription on Apple's backend, the
  honest answer for `transcribe` is neither "ready" nor a size — the OS may fetch a locale
  pack, and those bytes are shared with every app rather than charged to this one.

- **`coreai-doctor` — what will this app download.** `swift run coreai-doctor path/to/App`
  scans the sources for op calls and totals the models behind them, counting a model shared by
  three ops once. It asks `CoreAI.Op` rather than reimplementing the mapping, which is what
  keeps it right when a default changes — and the first thing it found was that `watch`,
  `watchDepth` and `scan` were missing from that enum, so an app using the live camera was
  being reported as having nothing to download.

- **`ModelStore.remoteSize(of:)` / `localSize(of:)` / `isCached(_:)` / `downloadPlan(for:)`** —
  sizes measured from the Hub rather than typed by hand, and the file list an adopter needs to
  hand to Apple's Background Assets. A Swift package cannot ship the downloader extension that
  framework requires; it can say exactly what to enqueue, which
  [`docs/BACKGROUND_ASSETS.md`](docs/BACKGROUND_ASSETS.md) walks through.

- **Errors for the three failures a shipped app actually hits** — `insufficientStorage`,
  `unsupportedDevice`, `systemAssetsUnavailable`. Every existing case was the vocabulary of
  someone who knows how a model repository is laid out (`notAHuggingFaceRepo`,
  `variantNotFound`), and none of them fire on the developer's machine, where the device is
  supported, the disk has room and the model is cached from the last run.

### Fixed

- **`Examples/OpsGallery` no longer built.** `watch`, `watchDepth` and `scan` joined
  `CoreAI.Op` (see `coreai-doctor` above) and the gallery's three switches over the enum were
  never extended, so the app stopped compiling. The grid now renders `CoreAI.Op.gallery` —
  everything except the streaming ops, which need a camera screen rather than a Run button and
  have their own examples — and the switches cover them explicitly, so the next streaming op
  cannot silently add a card whose button cannot work.

- **Every catalog VLM failed to load on iOS.** An iOS bundle built AOT holds one
  `*.aimodelc` directory and no `.aimodel`. `KitVisionModel` looked for `.aimodel` alone,
  found nothing, and fell back to a conventional `<name>.aimodel` — handing the runtime the
  path of a file that was never published (`Asset … is malformed: Missing hash file`). The
  only VLM that ran on a phone was Apple's own `SystemLanguageModel`.

  The same resolver had been written five times across the kit, and three copies were wrong:
  the ASR and audio encoders carried the identical fallback, and `KitDocReader` did not know
  `.aimodelc` at all. Those three had no iOS AOT bundle published yet, so they were waiting
  rather than working. All five now share `GraphBundle.resolve(in:)`, called inside
  `VLRuntime` / `ASRRuntime` / `AudioRuntime` so a new caller cannot miss it. AOT still wins
  over JIT when a bundle carries both — picking JIT pays on-device specialization every cold
  start, which is the cost the AOT build exists to remove.

- **A supported language in an unsupported region no longer hides transcription.** A device
  set to English-in-Japan reports `en_JP`, which is in nobody's supported-locale list, while
  ten English locales sit installed — strict matching declared the feature unsupported on a
  machine where it plainly worked. Resolution now falls back to the language, preferring the
  requested region, then the language's likely region via ICU (`en` → `en_US`), then anything
  installed. A language Apple genuinely cannot do is still refused rather than silently
  swapped for another. Found the first time this ran on a real machine.

### Changed

- **Transcription runs on Apple's on-device transcriber by default, at no download cost.**
  `CoreAI.transcribe` used to pull Whisper — 3.2 GB on an iPhone — for a capability iOS 27
  ships in `SpeechAnalyzer` + `SpeechTranscriber`, with OS-managed locale assets shared
  across apps. This package's own porting contract has a gate for exactly that ("Apple's
  stock stack does not already ship this capability. If it does, stop") and speech-to-text
  was the clearest case of it in the catalog. Measured on this machine: 45 supported locales,
  ten already installed, a 3.9 s clip transcribed in 0.18 s.

  `options: .model("whisper-large-v3-turbo")` opts back in, for a locale Apple lacks on the
  device in hand, behaviour that must not change under an OS update, or an offline guarantee.

- **`CoreAI.transcribeMeeting` is now 238 MB rather than 3.4 GB.** Who spoke when is the one
  speech capability Apple ships nothing for — `Speech.framework` has no speaker API at all —
  so Sortformer is a real download and the transcription it is paired with is free. The new
  `SpeechToText` protocol is the seam: `SystemTranscriber` and `KitTranscriber` are
  interchangeable, and `MeetingTranscriber(asr:)` takes either.

  Text-to-speech is deliberately **not** changed. Apple's voices are free but cannot be
  cloned, so the catalog TTS models still do something the system cannot, and defaulting them
  away would leave a wrapper with no reason to exist.

### Added

- **`VoiceActivityDetector`** — where speech starts and stops, which the speech design calls
  the only genuinely new component behind `listen()`. Everything else in the live speech path
  is composition of models that exist; deciding that a pause ended an utterance is not
  something a model does, and Apple's nearest offering (`SNClassifySoundRequest`) is a
  classifier over second-long windows, not an endpointer. Energy against an adaptive noise
  floor, with hysteresis, a hangover across stop consonants, and `minimumSilence` exposed
  because it is the latency choice and there is no correct value for it. No model, no
  dependency. `segments(in:)` is useful with no microphone in sight: it cuts a two-hour
  recording at pauses, which is how you feed a model whose window is thirty seconds.

  Two things the tests changed. `minimumSpeech` was being applied to the *padded* clip, so a
  40 ms click became 440 ms of "speech" and passed the filter that exists to reject it — the
  padding is for the model's benefit and must not decide what counts. And an utterance could
  stay open indefinitely: an energy detector cannot tell a fan from a voice, so a step in room
  level held one open forever and what eventually reached the model was longer than its
  window. `maximumSpeech` bounds it, splitting mid-word rather than handing over a clip
  nothing can take.

- **`KitTracker`** — detections in, stable identities out. A detector is stateless by
  construction, so "is that the same person as last frame" is a question no model answers and
  Apple ships nothing for; every counting, dwell-time or follow-the-object feature needs it
  and rebuilds it. Hungarian association (optimal, not greedy — greedy swaps identities
  exactly when two objects cross), constant-velocity prediction against real elapsed time,
  and a second association pass over low-confidence detections so a partly-occluded object
  keeps its id. No model, no framework, 18 tests.

  Two bugs the tests caught, both of which would have shipped: assignment crashed whenever
  there were more tracks than detections — which is every frame where something leaves — and
  IoU-only association fell apart at low frame rates. The second one matters most: box
  overlap between consecutive frames drops from 0.82 at 30 fps to 0.25 at 5 fps for the same
  motion, and 5 fps is what this package's own thermal governor produces on a hot phone. It
  worked on the desk and would have assigned a new id to every object in the user's hand.
  Association now falls back to centre distance, gated by how far the object could actually
  have travelled in the elapsed time.

- **`CoreAI.watch()` / `watchDepth()` — the camera as an op.** `CameraFeed` has vended frames
  since the beginning and no op consumed them, so every live example wrote its own two-stage
  task pipeline, stale-frame policy and stats window. That loop is now `LivePipeline`
  (CoreAIKitCore, Foundation only) with `LiveVision` over it, and `CoreAI.watch()` on top:
  detections per frame, normalized boxes, `result.stats` carrying *measured* frame rate,
  median latency, dropped frames and thermal state. Attach an `AVCaptureVideoPreviewLayer`
  to `watch.captureSession` and the preview costs nothing.

- **Thermal governor, on by default.** A sustained camera-plus-model loop is the hottest
  thing an app can do on a phone. `LiveGovernor` halves the target frame rate at a
  `.serious` thermal state and quarters it at `.critical` — `thermalBackoff: 1` opts out,
  which is appropriate only for a bench run.

- **Event triggers — `CoreAI.watch(for:)`.** A small detector runs continuously and decides;
  the expensive model runs on the frames that matter. `.label("person")`, `.anything()`, or
  `.when { … }`, each carrying a cooldown, because a predicate over a live stream is true for
  as long as the object is in shot. Yields a rendered `CGImage` only when it fires.

- **`KitDetector` real-time path** — `prepare(_:)` / `detect(_:)` over a 32BGRA capture
  buffer, plus `inputSize`. `Examples/DetectCamera` hand-wrote an enum to put RF-DETR and
  YOLOX behind one prepare/detect surface; that abstraction now lives in the package where
  it belongs.

- **`CoreAI.scan(videoAt:)` — a video file as a timeline.** `recognizeAction` answers one
  question about a whole clip; an app holding an hour of footage usually wants to know
  *where*. `scan` samples, runs a model per sample, and stamps the results; `scan(videoAt:for:)`
  is the offline twin of `watch(for:)`, with the cooldown counted in video time. Unlike the
  live path a scan drops nothing — it was asked for a specific set of samples and delivers
  all of them.

- **`VideoFile.stream` picks its reader from the sample rate.** Seeking costs per sample,
  sequential decode costs per clip, and the two cross over near one sample per second of
  30 fps source. Measured on an M4 Max over a 60 s clip: at 15 samples/s sequential is
  **17× faster** (0.48 s vs 8.20 s), at 0.1 samples/s seeking is **4.2× faster** (0.11 s vs
  0.47 s). `.automatic` chooses; `.seeking` / `.sequential` override. The sequential path
  rides OS 27's `AVAssetReader.outputProvider`, so it suspends rather than blocking a thread.

- **Scene-change sampling.** `minimumChange` skips frames too similar to the last one kept
  (mean absolute difference on a 32×32 grey thumbnail). On an 8 s clip that is a slow zoom
  over one photo, 2 samples/s runs the detector 16 times and `--changes` runs it once.

- **`Examples/LiveCamera`** — the four live tasks as four tabs on an iPhone, with the
  measured stats and the thermal governor visible on screen, plus `swift run live-cli` for
  the offline half (video scan and the preprocessing benchmark) with no device.

### Changed

- **Parakeet is no longer described as something Apple does not have.** `apple/coreai-models`
  merged its own Parakeet TDT 0.6B v3 export on 2026-08-07 (#136) and live streaming over it
  on 08-21 (#184), in a package that builds for iOS as well as macOS. Three places here —
  `docs/TASK_MAP.md`, `docs/SPEECH_API.md` and `Examples/Transcribe/README.md` — argued from
  a macOS-only Parakeet, and one of them called an iOS port "the cheapest high-value work on
  the map". Corrected in place with the date, rather than deleted: the catalog entry is
  still a macOS-only bundle and that is still worth saying out loud. Kokoro is the half of
  that argument that survives.

### Fixed

- **0.3.0 does not compile against the current Xcode 27 SDK; this release does.**
  FoundationModels renamed `LanguageModelCapabilities.init(capabilities:)` to `init(_:)`
  between Xcode 27 beta 3 and beta 5, which is five call sites here and one in the model
  fork. `docs/STABILITY.md` tells adopters to depend on a tag because shipping apps depend
  on this package — so between 2026-08-16 and today, the documented way to adopt CoreAIKit
  was the one way that failed, and it failed at compile time with an error naming Apple's
  type rather than anything of ours. `main` has been buildable since; nothing was tagged.
  If you are pinned to `from: "0.3.0"`, move to `0.4.0`.

- **Three examples were resolving Apple's package, not the patched fork.** `DocChat`,
  `FMToolDemo` and `GuidedDemo` carried a `Package.resolved` pinning
  `apple/coreai-models` 0.1.0 — a lockfile from before the fork existed. Every example
  depends on the kit by path, so the kit's own `exact:` requirement won them back on the
  first resolve; nobody had run one. All 25 lockfiles now record `0.2.2-zoo`.

- **CI can no longer pass by pinning an Xcode nobody uses.** Every workflow checked that
  `$DEVELOPER_DIR` was a directory and nothing else. At GA that check stops meaning
  anything: the release Xcode installs alongside the beta, the beta's folder keeps
  existing, and the gates keep publishing green for an SDK no shipping app is built with.
  `.xcode-pin` now names path *and* ProductBuildVersion in one place, and
  `scripts/check-xcode-pin.sh` refuses to run the job unless the installed build is the
  one the pin claims and no release build of the same Xcode train has appeared beside it.

- **Live depth no longer renders a bitmap per frame.** `LiveVision.depth` fed the estimator
  through `CIContext.createCGImage`; it now takes the capture buffer directly through the new
  `PixelBufferPreprocessor` (vImage scale + vDSP channel split, scratch reused). Measured on
  an M4 Max, 640×480 → 224², release build: **0.13 ms against 0.78 ms**, about 6×. The two
  paths agree bit-for-bit on flat colour, so this is not a change in what the model sees —
  only in what it costs to hand it over. `DepthEstimator` gains the matching
  `prepare(_:)` / `estimateDepth(_:)` real-time pair, and `ObjectDetector`'s previously
  private fast path is now the shared implementation.

- **Op models are now evictable.** `OpModels` and its six siblings cached every load and
  never released one, so an app calling three ops in sequence kept three models resident and
  was jetsammed on a phone — the capacity planning the op layer exists to remove. All
  thirteen caches now share `ResidentCache`, admitting loads through a process-wide
  `ModelResidency`: least-recently-used models are dropped to make room, a model an op is
  currently running on is never dropped, memory pressure drops everything idle, and a
  re-load after eviction is a cache miss rather than an error. `CoreAI.evictModels()` and
  `CoreAI.residentModels()` expose it.

## [0.3.0] — 2026-07-28

### Added

- **iOS dynamic-KV guard** — the engine (coreai-models `0.2.1-zoo`) now caps a pipelined
  turn's KV pre-grow at capacity 1024 on iOS for dynamically-sized-KV bundles: the on-device
  compiler miscompiles those specializations at seq ≥2048 (corrupt output from token 1).
  ChatDemo additionally clamps `maxResponseTokens` to 896 on iOS as belt-and-suspenders.
  Tracked in [#5](https://github.com/john-rocky/coreai-kit/issues/5); upstream
  apple/coreai-models#124. Both guards come out when the compiler fix lands.

- **Nanbeige4.2 3B** (`nanbeige4.2-3b`) — chat, the catalog's **first community-contributed
  model**: ported and published by [@ukint-vs](https://github.com/ukint-vs)
  ([zoo PR #6](https://github.com/john-rocky/coreai-model-zoo/pull/6)), served from the
  contributor's HF repo at a pinned revision. A recurrent Llama — 22 physical blocks execute
  twice per token (44 KV-cache layers) — int8, 4.6 GB, `thinking` chat template. Device-gated
  on iPhone 17 Pro: token-exact vs the fp32 oracle (24/24 ×2 runs), 8.5 prefill / 6.4 decode
  tok/s settled (two-pass models pay ~2× weight traffic per token; Mac M4 Max: 57 tok/s).

- **`KitSeparator`** — music source separation (Mel-Band RoFormer, Kim Vocal): a song in, a
  vocals stem and an instrumental stem out. `separate(_:)` on a decoded stereo mix or
  `separate(contentsOf:)` on any file AVFoundation reads. New catalog kind `separation`
  (`melband-roformer-vocal`). iPhone 17 Pro: an 8 s chunk in 1.23 s (~6.5x real-time).
- **`KitDialogue`** — multi-speaker / dialogue text-to-speech (VibeVoice-Realtime-0.5B), the first
  kit API that performs a *script*: `perform("Speaker 1: …\nSpeaker 2: …")` renders each turn from
  its own voice preset and concatenates them. 25 packaged voices, free text (Qwen2.5 tokenizer +
  mmapped fp16 embedding table — no torch, no 272 MB read). `KitSpeaker(catalog:)` also accepts
  the id and speaks one line in the default voice. Pairs with `KitDiarizer` for a
  generate → diarize loop.
- `AudioFile.pcmStereo(_:sampleRate:)` — decode any audio file to stereo at a chosen rate (the
  separation models need the full band, not the 16 kHz speech downmix).
- `StatefulGraphModel.seedState(keys:values:prefillLength:)` — seed a decode bundle's KV cache from
  an **external** prefill (a voice prompt, a cached prefix) instead of replaying it.

- **[`SECURITY.md`](SECURITY.md)** — the trust boundary written down: two hosts (GitHub raw for
  the catalog, Hugging Face for the weights), revision pins as the integrity mechanism, and the
  parts that are absent — `catalog.json` is unsigned, downloads carry no independent checksum
  manifest, bundles are not code-signed. Includes the escape hatch for production apps:
  `ModelCatalog.load(from:)` takes a URL, so you can host the catalog yourself.

- **[`AGENTS.md`](AGENTS.md)** — the contract for coding agents building on the kit: which layer
  to reach for (Apple's Foundation Models first), the catalog as data to read rather than guess
  at, the failures that only appear on a real device, and an accurate statement of what
  "verified" covers.

### Fixed

- **The offline fallback catalog resolved unpinned revisions.** `ModelCatalog.load()` falls back
  to the compiled-in `ModelCatalog.builtin` when the network fetch of `catalog.json` fails, and
  that literal carried no revision pins — while `CatalogEntry.modelID(path:)` resolves
  `revision ?? "main"`. A network failure therefore downgraded a shipped app from the exact bytes
  that were gated to whatever the model repository's `main` branch held at that moment, which is
  the opposite of what the catalog documents. Pins are now generated from `catalog.json` into
  `BuiltinPins.swift` and overlaid onto the literal; CI fails if the two drift, and
  `BuiltinPinsTests` fails if any entry reaches `builtin` without a pin. **Apps pinned to 0.2.0
  or earlier are affected** — the fallback path is the one that matters here, so an app that has
  never seen a catalog fetch failure has never hit it.

## [0.2.0] — 2026-07-10

### Added

- **Catalog revision pinning.** Every `catalog.json` entry now carries a `revision`
  (a Hugging Face commit hash), and every `(catalog:)` initializer downloads from
  exactly that revision — including multi-bundle models (VL towers, TTS glue, OCR
  assets, Gemma PLE tables, ASR encoder pairs). A push to a model repo can no longer
  change what apps receive; pins are bumped deliberately by re-running
  `scripts/pin-catalog.py` and committing the catalog change.
- `CatalogEntry.revision`, `CatalogEntry.modelID(path:)`, and `pinned(_:)` on
  `ModelID`, `GemmaModelID`, and `VLModelID`. Older catalogs without pins keep
  resolving `main`; older package versions ignore the new field — the change is
  backward and forward compatible.
- `scripts/pin-catalog.py` — re-pins all entries to each repo's current head
  (`--check` verifies every entry is pinned and reports drift, for CI).
- CI: build + unit tests on every push/PR, plus a nightly gate that fetches the live
  catalog, downloads a pinned model, and generates on the GPU end-to-end
  (self-hosted runner, macOS 27 beta + Xcode 27 beta).

### Fixed

- macOS build with current Xcode 27 betas: `os_proc_available_memory()` is now
  compiled only on iOS — the newer SDK marks it unavailable on macOS, which broke
  `swift build` for Mac consumers. Semantics are unchanged (macOS always owned
  tables).
- `catalog.json` / built-in snapshot drift: `nemotron-3.5-asr-streaming-0.6b` was
  missing from the built-in snapshot, while `mineru2.5-pro`, `glm-ocr`, and TimesFM's
  `static-shape` engine hint were missing from the shipped `catalog.json`. The sync
  test now pinpoints per-entry drift and requires every shipped entry to carry a pin.

### Changed

- `KitASRModel`, `KitWhisperModel`, `KitParakeetModel`, and `KitForecaster`
  `(catalog:)` initializers now consult the live catalog like every other surface, so
  unknown ids and wrong-platform requests fail with the same clear, early errors.
- First run on 0.2.0 re-downloads models once: pinned bundles cache under their
  revision (`<repo>/<revision>/<variant>`) instead of `main`.

## [0.1.0] — 2026-07-07

Initial tagged release: `ModelStore` + `ModelCatalog`, `ChatSession` (streaming,
stats, guided generation), FoundationModels providers (text / VL / Gemma), vision
pipelines (CLIP, depth, detection, super-resolution, video), embeddings + retrieval,
ASR / TTS / diarization / OCR / audio-QA / diffusion-LM / forecasting surfaces, and
SwiftUI components.
