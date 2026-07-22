# OpsDemo

The anchored ops end to end. Point it at a voice memo and one audio file becomes a
transcript, a cleaned-up text, a one-line summary, typed action items, and a Japanese
translation — all on device. No sessions, no prompts, no model names in app code: ops are
the stable API and the kit resolves a catalog model behind them.

## Run

```bash
swift run OpsDemo                    # text ops on built-in samples (qwen3 4B)
swift run OpsDemo voice-memo.wav     # full pipeline (downloads Whisper large-v3-turbo once)
```

Real pipeline output (Mac, ~38 s warm for all five ops):

```
> CoreAI.transcribe(memo)
[transcript] Team meeting notes. Please ship 12 units of the Alpha widget to the Osaka
office by Friday. Also, schedule a follow-up call with Dana next Tuesday to review the
launch plan.

> CoreAI.summarize(clean, style: .oneLine)
[summary] Ship 12 Alpha widgets to the Osaka office by Friday and schedule a follow-up
call with Dana next Tuesday to review the launch plan.

> CoreAI.extract(clean, as: ActionItems.self)
[task] Please ship 12 units of the Alpha widget to the Osaka office by Friday.
[task] Schedule a follow-up call with Dana next Tuesday to review the launch plan.

> CoreAI.translate(clean, to: .japanese)
[ja] チーム会議のメモ。12個のアルファウィジェットを大阪支社に金曜日までに発送してください。…
```

`extract` shapes the reply with the `@Generable` type's generation schema and parses it
back through the framework, so the demo's `ActionItems` / `Order` types are exactly what
you would write for Apple's `respond(generating:)`. Override any op's model per call,
e.g. `options: .model("qwen3-0.6b")` when speed matters more than fidelity.
