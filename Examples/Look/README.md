# Look — ask about a photo, answered on the device

Take a photo, pick one or tap an example, type a question about it, and a vision-language model on
your iPhone answers in text. Point the phone at a sign, a menu, a device or a view and ask what you
want to know: the model on the phone starts answering within seconds and answers faster than you can
read, and the photo never leaves the phone. One screen, one photo and one question in, one answer out;
the line under the answer says when its first token came, how fast the rest followed, and where it
ran. The photo and the question go to the model in the app's own process.

The model is `minicpm-v-4.7-1b` from the kit's catalog: MiniCPM-V 4.7 1B, Hugging Face
`mlboydaisuke/MiniCPM-V-4.7-1B-CoreAI` at revision `51278afd`, an int8 decoder and an int8 vision
tower that run on the GPU (`gpu-pipelined`), 1.94 GB. The take-home is
[`Sources/QuickStart.swift`](Sources/QuickStart.swift): one function, a photo and a question in, the
answer out.

## Run it with your own photo

```sh
cd Examples/Look
tools/fetch_examples.sh              # the three example photos (CC0), checked by size and sha256
export DEVELOPMENT_TEAM=XXXXXXXXXX   # your team id; or pick your team in Signing & Capabilities
xcodegen generate                    # brew install xcodegen
open Look.xcodeproj                  # choose your iPhone (iOS 27) and Run; the scheme runs Release
```

1. The app loads the model as it starts. The first launch downloads it from Hugging Face (1.94 GB,
   the catalog row `minicpm-v-4.7-1b` at `51278afd`); after that it loads from the phone. READY
   means it can answer.
2. Take a photo, Pick a photo, or tap one of the three examples; tap a question under the box, or
   type your own; Ask.
3. On an iPhone 18 Pro (iOS 27.2), the first launch after the install reached READY 33 s after it
   started (the model is specialized for the phone once, after the download) and its first question
   showed the first token 3.5 s after Ask. From the next launch on, READY came 2.9 s after the start
   and the first token 0.99 s after Ask. These numbers were measured on 2026-10-15 with the previous
   build of this sample; the layout of this build is still to be checked on a phone.

The app needs the entitlement `com.apple.developer.kernel.increased-memory-limit`
(`Look.entitlements`): the vision tower specializes beside the loaded decoder, and without the
higher limit that first specialization died with `std::bad_alloc` on an iPhone 18 Pro (VLChat, the
same model). Run it in Release: the engine's per-token host work is about 3× slower unoptimized.
The same project also builds for a Mac (macOS 27, Apple silicon), with Pick a photo and the examples
and without Take a photo; the numbers on this page are from an iPhone, and Look has not been run on
a Mac.

## The screen

Top to bottom:

1. the state — LOADING, READY, THINKING, DONE, STOPPED, FAILED — and what is happening;
2. your photo, whole (fitted into its row, its aspect kept);
3. the three examples, Pick a photo and Take a photo;
4. the question box and three example questions (a tap puts one in the box, which stays editable);
5. Ask and Stop;
6. the answer as it streams, in 24 pt type;
7. the numbers line, in two lines;
8. the model, its revision, the bundle, the device, the OS and the app version.

Everything sits in one band: the screen's width × 16/9, from the top of the safe area (402 × 714 pt
on an iPhone 18 Pro, under the status bar). The rows have fixed heights and the answer box gets the
rest: ≈ 234 pt on that phone by the rows' heights (not yet measured there); the photo's row is a
fifth of the band. A recording crops the same band, so the screen you use is the screen a recording
shows.

The model writes Markdown, and the screen renders inline markdown: `**bold**` in bold, a `#` heading
as a bold line, a `- ` item as `• `; line breaks inside a paragraph are shown as spaces, and a blank
line stays a gap between paragraphs; `>` quote marks and `---` rules are dropped. The record keeps the
text as written. Asked what the sign example says, the model writes the sign's words one per line in
bold (`**Please␣␣⏎KEEP␣␣⏎OFF␣␣⏎THE GRASS**`, the same in Hugging Face fp32); on the screen they read
as one bold phrase, "Please KEEP OFF THE GRASS".

An answer that fits the box shows from its first line to its last. A longer one — "What is this and
how do I use it?" runs to 200–280 tokens — follows its last line as it streams, and scrolls inside
the box.

Pick a photo opens the system photo picker, which runs outside the app and needs no permission. Take
a photo opens the camera and asks for the camera permission the first time. Every question goes to a
new session with the photo before it: one turn, no history.

## The numbers line

```
first token 0.99 s · 101 tok/s
GPU · on-device, offline
```

- first token: from the Ask tap to the first streamed snapshot that carries text. The model loads
  before READY and is not in it.
- tok/s: (answer tokens − 1) / (end of the stream − first token). The tokens are the session's own
  count, `session.usage.output.totalTokenCount`.
- snapshots are not tokens: the stream delivers text snapshots, and the record keeps their count
  (`snapshots`, and `text_updates` for those that changed the text) beside `output_tokens`. For the
  sign example's answer they were 44 and 41 against 42 tokens (the previous build, 2026-10-15); the
  line always uses the tokens.
- GPU: the bundle the catalog row names lives under `gpu-pipelined/`.
- offline: shown only while the device has no network path at all (`NWPathMonitor` reports none
  satisfied); with Wi-Fi or cellular on, the line says `on-device` only.

The line has two lines by design — the measured numbers, then where it ran — so it keeps 22 pt bold
on a phone: as one line it is 537 pt wide (laid out on a Mac), and an iPhone 18 Pro's column is
370 pt.

## What one question sends

1. The photo, as the screen got it: the image file decoded with ImageIO (JPEG, HEIC, PNG, WebP …;
   the first image, its EXIF orientation read; a long side over 4096 px is subsampled while it
   decodes), or the camera's image with its orientation.
2. A new `LanguageModelSession` on the loaded model and one prompt: `Attachment(cgImage,
   orientation:)`, then the question, as typed (trimmed). The app adds no prompt of its own.
3. The kit turns the image upright and draws it stretched into a 448 × 448 sRGB canvas
   (`VLImagePreprocessor`), the vision tower's input.
4. Greedy decoding (no temperature), at most 256 answer tokens (`GenerationOptions(maximumResponseTokens: 256)`).

## Autoplay and the record

For recordings and checks from a script, the app taps its own controls — the same functions the
thumbnails, the example questions and Ask call — when started with launch arguments:

```sh
xcrun devicectl device process launch --device <udid> com.coreaikit.look -- \
  -autoplay 1 -example sign -chip 0 -delay 2 -gap 2 -frames 1 -log 1
```

| argument | meaning |
|---|---|
| `-autoplay 1` | run the sequence once: wait for READY, then the trigger, `delay`, tap the example, `gap`, tap the question, `gap`, Ask |
| `-log 1` | append the steps to `Documents/look-autoplay.log` and write `Documents/look-run-<epoch>.json`; without `-autoplay 1`, record each question asked by hand (or by the UI test) the same way |
| `-model <id>` | a `vlm` catalog id (default `minicpm-v-4.7-1b`) |
| `-example sign` | `sign`, `object` or `scene` |
| `-chip 0` | the example question to tap (0, 1, 2); `-question "…"` types one instead |
| `-delay 2`, `-gap 2` | seconds before the example, and between the steps |
| `-trigger <file>` | wait until this file exists in Documents (a recorder creates it once it is rolling) |
| `-frames 1` | write the screen as PNGs to `Documents/frames/<epoch>/`: READY, the pick, the question, the first token, DONE (by hand: the first token and DONE) |
| `-quit 1` | exit after the record |
| `-probe 1` | log only START (the thermal state, the free bytes) and DONE; with `-quit 1`, exit |
| `-take 1` | kept in the record as `take`; the layout is the same without it |

The log has one line per step: START, READY, PICK, CHIP (or QUESTION), ASK, FIRST, DONE, or ERROR
with the reason, and INACTIVE / ACTIVE when the app stops and starts being the active app. The record
holds the question and the photo; the load and step times (`engine_ready_s`, `t_ready_s`, `t_ask_s`);
`first_token_s`, `generate_s`, `input_tokens`, `output_tokens`, `snapshots`, `text_updates`, `tok_s`;
the answer as written (`answer`) and as shown (`answer_shown`); the numbers line and the footer; the
device, the OS and the thermal state at the start and the end; the band in pixels (`take_rect_px`) and
the answer box in points (`answer_box_pt`: content, container, offset — the answer showed whole when
content ≤ container at offset 0); and `inactive_count`, `inactive_s`, `inactive_at`: the stretches when
the app was not the active app — a system alert over it (another app's permission request), Control
Center, another app in front — counted from the scene phase, so a recording can be checked for them.
An alert that is already on the screen when the app starts keeps it from ever becoming active; for
that case the record also holds `scene_first_active_s` (when the app first became active, null if it
never did) and `scene_at_record` (the phase when the record was written).

## The UI test

`UITests/LookUITests.swift` drives the screen through its own controls on an iPhone: it waits for
READY (up to 240 s, the first launch included), taps the sign example, the first example question and
Ask, waits for DONE (90 s), and attaches the numbers line and the answer to the test result. The app
runs with `-log 1`, so it writes its record too.

```sh
xcodebuild test -project Look.xcodeproj -scheme Look -destination 'platform=iOS,id=<udid>'
```

The first UI test on a phone asks for the phone's passcode on the phone.

## Example photos

Three photos, each CC0 1.0 according to its Openverse record; none shows a person, a brand or a logo.
They are not in the repository: `tools/examples.json` names each one's URL, size and sha256, and
`tools/fetch_examples.sh` downloads them into `Examples/` (the folder the app bundles) and checks both
numbers; `--check` checks without the network. When a URL stops serving the same bytes, the script says
so, writes nothing, and names the Openverse record and the source page.

| id | shows | file | source page |
|---|---|---|---|
| `sign` | a white sign on a metal post in a lawn: "Please" above "KEEP OFF THE GRASS" | `sign.webp`, 1024 × 768 (the URL ends in .jpg; the server sends WebP) | [rawpixel](https://www.rawpixel.com/image/6038398/please-keep-off-the-grass-sign) |
| `object` | light wooden chess pieces on a worn wooden chessboard, close up | `object.jpg`, 960 × 640 | [StockSnap](https://stocksnap.io/photo/chess-board-NUAITIXILP) |
| `scene` | a white lighthouse with a red base on rocks by the sea, dune grass in front | `scene.jpg`, 1024 × 683 | [Flickr](https://www.flickr.com/photos/66741228@N05/33074457825) |

## When it does not answer

| what happens | what the screen says |
|---|---|
| Ask without a photo | Choose a photo first: tap an example, pick or take one |
| Ask with an empty question | Type a question, or tap one under the box |
| the model does not load (no network on the first launch, a full disk) | FAILED, and the error's own description |
| a picked photo the app cannot decode | Could not read that photo |
| a very large photo | nothing: a long side over 4096 px is subsampled while it decodes |
| a photo stored sideways with an EXIF orientation | nothing: the screen shows it upright and the model gets it upright |
| a build without the example photos | The example photos are not in this build: see the README (the tiles show their names) |
| Stop during an answer | STOPPED, with the answer so far; the record says `stop: stopped` |

These are the app's own sentences, from `LookModel.swift`.

## Models

`-model <catalog id>` loads another `vlm` row of the catalog the same way (`KitVisionModel(catalog:)`):
`qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` (Mac only), `holo2-4b`, `minicpm-v-4.6`, `lfm2.5-vl-450m`,
`lfm2.5-vl-3b`, `north-micro-vision`. Look has been run with `minicpm-v-4.7-1b` only; the others are
untested in this app. The numbers line's backend and the footer come from the row that loaded.
