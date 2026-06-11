# FMToolDemo

Local tool calling behind Apple's `LanguageModelSession`: a Core AI bundle answers a
weather question by calling a Swift `Tool` — the model decides to call, the
FoundationModels framework executes it, and the answer is grounded on the result.
All on-device.

## Run

```bash
swift run -c release FMToolDemo                 # downloads qwen3 0.6B on first run
swift run -c release FMToolDemo /path/to/bundle # or use a local bundle
```

Tool calling needs a ChatML-speaking model (qwen3 family). Expect the tool call in the
transcript dump:

```
> What's the weather in Sapporo right now?
  [tool] get_weather(city: Sapporo)
[answer] It's sunny and 24°C in Sapporo right now.
[transcript]
  instructions / prompt / toolCalls: ["get_weather"] / toolOutput / response
```
