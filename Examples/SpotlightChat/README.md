# SpotlightChat

Local RAG with Apple's **`SpotlightSearchTool`** (WWDC26) driven by **your own model**. The
search tool is a plain `FoundationModels.Tool`, so it rides behind any `LanguageModel` — here a
Core AI zoo bundle via `KitLanguageModel`, not the system model. Everything runs on device.

The realistic on-device RAG shape, all behind one third-party model:

1. **`spotlight_search`** (Apple's system tool) finds candidate notes in the Core Spotlight index.
2. **`fetch_note`** (this app's own tool) reads a note's full body from the app's store.
3. The model grounds its answer in the text it read.

## Run

```bash
swift run -c release SpotlightChat                       # downloads qwen3-4B on first run
swift run -c release SpotlightChat --ask "What did I write about the Granite Pass attempt?"
swift run -c release SpotlightChat --system              # SystemLanguageModel baseline
swift run -c release SpotlightChat --model /path/to/bundle  # flagship zoo bundle (see note)
```

Expected (default question, `qwen3-4B`) — **abridged**: the real run also prints `[index]`
setup lines, first-run download progress, and a full `[transcript]` dump, and the
`[spotlight]` stream interleaves with the model's tool calls rather than in this fixed order.
The salient lines are:

```
> What did I write about the night hike?
  [spotlight] items[5] label=night hike: ... | Pine Hollow night hike | ...
  [fetch_note] note-003
  toolOutput Pine Hollow night hike: First night hike of the season. Headlamp died halfway …
[answer] You wrote about Pine Hollow — your first night hike of the season; your headlamp
         died halfway, so pack spare batteries next time.
```

## What this demonstrates

- **Apple's `SpotlightSearchTool` works behind a third-party `LanguageModel`.** The only
  capability needed is `.toolCalling` (the kit declares it for ChatML models). The tool's query
  schema is rendered into the prompt and the model emits a tool call the framework executes.
- **`tool.searchResults`** — an async stream of `SearchReply`s you can render live, separate from
  the model's prose (the `[spotlight]` lines).
- **`CSSearchableIndexDelegate`** wired via `CoreSpotlightSource(searchableIndexDelegate:)` for
  index recovery + `searchableItems(forIdentifiers:)` hydration.
- **Two-tool orchestration** — the same small model chains `spotlight_search` → `fetch_note`.

## Notes & gotchas (27.0 beta)

- **The search tool returns index *metadata*, not the body.** Even with
  `CoreSpotlightSource(fetchAttributes: [.contentDescription, ...])`, the model sees only title,
  dates, type, and identifiers — not the note text. (A raw `CSSearchQuery` *does* return
  `contentDescription`; `textContent` is index-only.) That's why the answer is grounded through
  the companion `fetch_note` tool, mirroring real apps where the index is a finding aid and the
  full content lives in your store.
- **Guidance level is a token gate.** The default `.complete` guidance is ~13 k tokens — it
  overflows a 4 k-context model instantly. This example ships `.focused(.items)` + `.compact`.
- **Model choice + `/no_think`.** Tool calling needs a ChatML model, so the kit enables it for
  the qwen3 family. The default is **qwen3-4B** (the 0.6B is too small for this tool's rich
  schema). qwen3 is a *thinking* model, and with this big tool schema its chain-of-thought can
  run to the token cap — the framework then reports "ended without producing a response"
  intermittently. The instructions append **`/no_think`** to disable qwen3 reasoning, which makes
  the search→fetch chain fast and reliable; it's ignored harmlessly by non-qwen models.
- **Flagship zoo model.** `--model <qwen3.5-0.8B bundle>` runs the chain on a hybrid zoo bundle,
  but hybrid bundles need a `coreai-models` engine with hybrid KV-state support; the public stock
  engine the kit depends on rejects them. Plain-transformer catalog bundles run on stock.
