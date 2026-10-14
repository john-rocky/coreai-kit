# Look — ask about a photo, answered on the device

Pick a photo (or take one, or tap an example), type a question about it, and a vision-language model
running on your iPhone or Mac answers in text. The line under the answer says when its first token came,
how fast the rest followed, and where it ran. Nothing leaves the device: the photo and the question go to
the model in the app's own process.

The model is `minicpm-v-4.7-1b` from the kit's catalog (MiniCPM-V 4.7 1B, Hugging Face
`mlboydaisuke/MiniCPM-V-4.7-1B-CoreAI`: an int8 decoder and an int8 vision tower, run on the GPU, 1.9 GB).
The take-home is [`Sources/QuickStart.swift`](Sources/QuickStart.swift): one function, a photo and a
question in, the answer out.

## Run it

1. The three example photos are not in the repository. Put `sign.webp`, `object.jpg` and `scene.jpg`
   (CC0 1.0) into `Examples/` here before you build; without them the example tiles stay empty, and Pick a
   photo and Take a photo still work.
2. `open Look.xcodeproj`, pick your team in Signing & Capabilities (or `export DEVELOPMENT_TEAM=…` before
   `xcodegen generate`), and run on an iPhone with iOS 27 or a Mac with macOS 27. The scheme runs Release.
3. The app loads the model as it starts: the first time from Hugging Face (1.9 GB), after that from the
   device. READY means it can answer.
4. Tap an example (or Pick a photo, or Take a photo), tap a question under the box (or type one), and Ask.

## The screen

Top to bottom: the state (LOADING, READY, THINKING, DONE, STOPPED, FAILED) and what is happening; your
photo; the three examples, Pick a photo and Take a photo; the question box and three example questions;
Ask and Stop; the answer as it streams, in large type; the numbers line; the model, its revision, the
bundle, the device, the OS and the app version.

Pick a photo opens the system photo picker, which runs outside the app and needs no permission. Take a
photo opens the camera and asks for the camera permission the first time. A tap on an example question
puts it in the box, which stays editable. Every question goes to a new session with the photo before it:
one turn, no history.

The model writes Markdown. The screen sets `**bold**` in bold and drops the `#` and `- ` marks; the
record keeps the text as the model wrote it. The answer box shows the answer from its first line; an answer
taller than the box scrolls inside it.

## The numbers line

The line reads `first token <seconds> s · <rate> tok/s · GPU · on-device, offline`:

- first token: from the Ask tap to the first streamed snapshot that carries text. The model loads before
  READY and is not in it.
- tok/s: (answer tokens − 1) / (end of the stream − first token). The token count is the session's own
  (`session.usage.output.totalTokenCount`); the record keeps the number of streamed snapshots beside it.
- GPU: the bundle the catalog row names lives under `gpu-pipelined/`.
- offline: shown only while the device has no network path at all; otherwise the line says `on-device`.

## Autoplay

For recordings and checks from a script, the app drives its own buttons (the same functions a tap calls)
when started with launch arguments:

```sh
xcrun devicectl device process launch --device <udid> com.coreaikit.look -- \
  -autoplay 1 -example sign -chip 0 -delay 3 -gap 2 -take 1 -frames 1 -log 1
```

| argument | meaning |
|---|---|
| `-autoplay 1` | run the sequence once: wait for READY, then the trigger, `delay`, tap the example, `gap`, tap the question, `gap`, Ask |
| `-model <id>` | a `vlm` catalog id (default `minicpm-v-4.7-1b`) |
| `-example sign` | `sign`, `object` or `scene` |
| `-chip 0` | the example question to tap (0, 1, 2); `-question "…"` types one instead |
| `-delay 3`, `-gap 2` | seconds before the example, and between the steps |
| `-trigger <file>` | wait until this file exists in Documents (a recorder creates it once it is rolling) |
| `-take 1` | lay the whole screen out in a 9:16 band under the status bar |
| `-frames 1` | write the screen as PNGs at READY, the pick, the question, the first token and DONE |
| `-log 1` | append the steps to `Documents/look-autoplay.log` and write `Documents/look-run-<epoch>.json` |
| `-quit 1` | exit after the record |
| `-probe 1` | log only START (the thermal state, the free bytes) and DONE; with `-quit 1`, exit |

The record holds the question, the photo, the load and step times, the first token, the token and
snapshot counts, the rate, the answer as written and as shown, the numbers line, the device, the OS,
the thermal state, the screen and take band in pixels, and the answer box's geometry.

## Notes

- Entitlements (iOS): `com.apple.developer.kernel.increased-memory-limit` (`Look.entitlements`): the
  vision tower specializes beside the loaded decoder.
- Run in Release: the engine's per-token host work is about 3× slower unoptimized.
