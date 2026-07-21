# OpsDemo

The two anchored ops end to end: `CoreAI.summarize` turns a paragraph into a one-line
summary, `CoreAI.extract` pulls a typed `@Generable` value out of an email. No sessions,
no prompts, no model names in app code — ops are the stable API and the kit resolves a
catalog model behind them.

## Run

```bash
swift run   # downloads qwen3 0.6B on first run, then cached
```

Expected shape of the output:

```
> CoreAI.summarize(article, style: .oneLine)
[summary] Apple's FoundationModels framework allows apps to ...

> CoreAI.extract(email, as: Order.self)
[order] product=AlphaWidget quantity=12 city=Osaka
```

`extract` runs guided generation on the sequential engine, so the result parses by
construction. Override the model per call with `options: .model("qwen3-4b")`.
