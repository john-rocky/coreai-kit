# DocChat

On-device RAG, end to end: your `.md`/`.txt` notes are indexed with EmbeddingGemma
embeddings, and a local LLM answers questions through `LanguageModelSession` with a
retrieval `Tool` — the model decides when to search, the framework executes the search
and grounds the answer. Nothing leaves the machine.

## Run

```bash
swift run -c release DocChat ~/notes "What do my notes say about the bike trip?"
```

First run downloads EmbeddingGemma (~600 MB) and qwen3 0.6B (~350 MB); both cache.
Local bundles can be supplied via `KIT_EMBED_BUNDLE` / `KIT_CHAT_BUNDLE`.

Expected shape:

```
indexed 12 chunks from 3 files
> What do my notes say about the bike trip?
  [tool] search_notes("bike trip") → bike-trip.md 0.71, …
[answer] Your notes describe a weekend ride along the coast …
```
