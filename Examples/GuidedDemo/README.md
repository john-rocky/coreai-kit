# GuidedDemo

Guided generation (constrained decoding): per-step logits are masked through a JSON
schema's grammar (xgrammar bitmask) before sampling, so the model's output is valid
JSON for the schema **by construction** — no retries, no parse failures. All on-device.

Two paths, same bundle:

1. `ChatSession.respond(to:generating:schema:)` — JSON schema string in, `Codable` out.
2. `LanguageModelSession.respond(to:generating:)` with a `@Generable` type —
   FoundationModels derives the schema and parses the result.

## Run

```bash
swift run -c release GuidedDemo                 # downloads qwen3 0.6B on first run
swift run -c release GuidedDemo /path/to/bundle # or use a local bundle
```

Expected shape:

```
> Give facts about the capital of Japan.
[typed] Tokyo, Japan — population 14000000
> Plan a short trip to Kyoto.
[plan] Kyoto in spring: Kinkaku-ji, Fushimi Inari, Arashiyama
```

## Notes

- Guided generation needs per-step logits, so both paths load the **sequential**
  engine (`engineVariant: .sequential`). The default pipelined engine samples on-GPU
  and rejects schema requests. Decode is slower than pipelined chat — fine for short
  structured output.
- Constrained turns can't think (`<think>` is not grammar-legal JSON), so thinking
  models answer directly.
