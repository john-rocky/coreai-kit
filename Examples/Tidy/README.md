# Tidy

Raw dictation in, written text out — **S1-mini by Superwhisper** on device. Fillers dropped,
false starts resolved to whatever the speaker landed on, punctuation and capitalization
applied, and spoken numbers, dates, times, currency and email addresses written the way a
person would type them.

```
so um i need to like send the the report by uh friday no wait make that thursday
→ I need to send the report by Thursday.
```

This is the other half of the dictation path: `Examples/Transcribe` (or Apple's own
transcriber) produces the raw transcript, and this turns it into text someone would send.

## Run

```bash
# GUI (iPhone or Mac)
xcodegen generate
open Tidy.xcodeproj

# headless (macOS), the same function the GUI calls
swift run tidy-cli --text "so um i need to like send the report by uh friday no wait thursday"
swift run tidy-cli --file meeting.txt --styling formal --structure lists
```

## The three controls

They are the model's own trained axes — there is no free-text instruction:

| | values | what it changes |
|---|---|---|
| `--styling` | `casual` / `semi-casual` / `semi-formal` / `formal` | register: `hmm im gonna be late` stays as spoken, or becomes `I am going to be late.` |
| `--structure` | `prose` / `lists` | an enumeration in the speech becomes a Markdown bullet list |
| `--context` | `general` / `email` | email shape: greeting, body and sign-off as their own blocks |

**English only** — the model has no language control.

**Filler-only input returns the empty string.** That is the model working, not a failure; the
CLI says so on stderr and keeps stdout empty.

## Long transcripts

`normalize` cuts input past ~450 tokens at word boundaries and stitches the rewrites, because
on iPhone the shipped engine caps prompt + generated at 1024 tokens: a whole meeting transcript
passed in one call stops mid-sentence rather than failing. Measured on iPhone 17 Pro — a
611-token transcript produced 413 tokens, every one correct, then stopped at absolute position
exactly 1024. Mac has no such cap. See the
[model card](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/s1-mini/README.md).

## Where the code is

- `Sources/QuickStart.swift` — the take-home: one typed function, no UI. The GUI and the CLI
  both call it.
- `CLI/main.swift` — argument shell over that function.
- `Sources/TidyModel.swift` / `TidyView.swift` — the display shell.

**Naming.** S1-mini's licence adds a term to Apache-2.0: any use, distribution or product
integration must keep identifying it as **"S1-mini"** by **"Superwhisper"**, with that exact
capitalization.
