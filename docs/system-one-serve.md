# System One, on device: the serve command

A typed decision — a text, a question with fixed answers, each answer's probability, nothing
generated — is what the hosted System One APIs sell by the call. The same call runs on the
device with a catalog model: `CoreAI.decide` in Swift (`CoreAI.systemOne` for the hosted
request and response forms), or `systemone serve` for everything else. It answers `POST /v1/systemone` in the hosted request and answer forms, so a client
written for the hosted endpoint is switched by its base URL and nothing else changes:

```bash
brew install john-rocky/tap/systemone && systemone serve     # http://127.0.0.1:8090/v1/systemone
brew services start systemone                                 # the same, kept running by launchd
export TYPESAFE_BASE_URL=http://127.0.0.1:8090               # the official SDKs read this; other clients have their own base-URL setting
```

`systemone` is one signed, notarized 21 MB binary with no Swift toolchain behind it. The first
`serve` downloads MiniCPM5 2B (2.7 GB) into `~/Library/Application Support/CoreAIKit/Models`;
after that `brew services start` answers `/health` 0.7–4.6 s later (M4 Max; the spread is
how much of the 2.7 GB is still in the file cache, and the model load is most of it). `systemone ask
--state "…" --noul "…"` is one decision from the shell (`--json` for the wire form),
`systemone models` says what can decide and what is downloaded. From source it is
`swift run -c release systemone serve`, or `decide-cli serve` in `Examples/Decide`.
`systemone mcp` is the same decisions as a Model Context Protocol server on stdio:
`claude mcp add systemone -- "$(brew --prefix)/bin/systemone" mcp` (Codex: `codex mcp add …`;
Cursor: `~/.cursor/mcp.json`) and the agent's own sessions get a `decide` tool.

MiniCPM5 2B answers a question in about 40–65 ms after the state is read once (M4 Max), and
its decisions track the published full-precision readout (141/144 argmax on the authored
fixture). Ten whole uses — a form that fills from a copied email, a contract checklist, a
folder sorter, a car the model drives, a CSV with the columns you ask for, a command guard
for a coding agent, context compression, as-you-type reading — are in
[`Examples/Decide`](../Examples/Decide), with the question shapes that read correctly on a 2B
model and the ones that did not. [docs/SYSTEM_ONE.md](SYSTEM_ONE.md) is the one-page map.
