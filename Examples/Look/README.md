# Look — ask about a photo, or have the camera watch for what you name, on the device

Take a photo, pick one or tap an example, type a question about it, and a vision-language model on
your iPhone answers in text. Point the phone at a sign, a menu, a device or a view and ask what you
want to know: the model on the phone starts answering within seconds and answers faster than you can
read, and the photo never leaves the phone. One screen, one photo and one question in, one answer out;
the line under the answer says when its first token came, how fast the rest followed, and where it
ran. The photo and the question go to the model in the app's own process.

The second screen, [Watch](#watch), turns the camera into rules you write in words: "a chessboard",
"a sign with words on it", "water rushing over rows of stones" — each with one thing to do when the
model sees it (a sound, a buzz, a spoken line, a timer, a count, a log line, or an effect: the picture
stops on the frame the model answered and the app edits it at the box the model gives for the thing, with a sound).
Press Start and the model checks one frame at a time; a rule whose answer turns yes fires its reaction.
Both screens use the same model, loaded once.

The model is `minicpm-v-4.7-1b` from the kit's catalog: MiniCPM-V 4.7 1B, Hugging Face
`mlboydaisuke/MiniCPM-V-4.7-1B-CoreAI` at revision `51278afd`, an int8 decoder and an int8 vision
tower that run on the GPU (`gpu-pipelined`), 1.94 GB. The take-home is
[`Sources/QuickStart.swift`](Sources/QuickStart.swift): one function, a photo and a question in, the
answer out.

## Run it with your own photo

```sh
cd Examples/Look
tools/fetch_examples.sh              # the three example photos (CC0); with ffmpeg, the example video and the effect sounds
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
   showed the first token 3.5 s after Ask. From the next launch on, READY came 0.9–2.9 s after the
   start and the first token 1.0 s after Ask. Measured on 2026-10-15: the previous build gave READY
   2.9 s and the first token 0.99 s; this build, on the recorded take (a new process, the first
   question), gave READY 0.9 s after the start, the first token 1.01 s and 98.8 tok/s for the sign
   example's 42-token answer. On that phone the sign answer fits its box (202 of 234 pt) and the
   221-token how-to answer scrolls to follow its tail.

The app needs the entitlement `com.apple.developer.kernel.increased-memory-limit`
(`Look.entitlements`): the vision tower specializes beside the loaded decoder, and without the
higher limit that first specialization died with `std::bad_alloc` on an iPhone 18 Pro (VLChat, the
same model). Run it in Release: the engine's per-token host work is about 3× slower unoptimized.
The project also has a Mac destination (macOS 27, Apple silicon), with Pick a photo and the examples
and without Take a photo or the camera source. This version, the Watch screen included, built for a
Mac on 2026-10-24 (Release, arm64, unsigned); Look has not been run on one. The numbers on this page
are from an iPhone.

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

Everything sits in one band: the screen's width × 16/9 (402 × 714 pt on an iPhone 18 Pro), under the
Ask | Watch switch at the top — or right under the status bar with `-take 1`, which hides the switch
for a recording. The rows have fixed heights and the answer box gets the rest: 234 pt on that phone
(the record's `answer_box_pt`, 2026-10-21); the photo's row is a fifth of the band. A recording
crops the same band, so the screen you use is the screen a recording shows. The switch is off while
an answer streams or a watch runs: the two screens share the model.

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
first token 1.01 s · 98.8 tok/s
GPU · on-device
```

- first token: from the Ask tap to the first streamed snapshot that carries text. The model loads
  before READY and is not in it.
- tok/s: (answer tokens − 1) / (end of the stream − first token). The tokens are the session's own
  count, `session.usage.output.totalTokenCount`.
- snapshots are not tokens: the stream delivers text snapshots, and the record keeps their count
  (`snapshots`, and `text_updates` for those that changed the text) beside `output_tokens`. For the
  sign example's answer they were 44 and 41 against 42 tokens (the previous build, 2026-10-15; the
  recorded take of this build: 44 snapshots, 42 tokens); the line always uses the tokens.
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

## Watch

Watch turns the camera into rules you write in words. A rule is a thing to look for — "a chessboard",
"a sign with words on it", "a coffee mug" — and one thing to do when the model sees it. Press Start
and the app shows the model one frame at a time; when a rule's answer turns yes, its reaction fires
and its symbol shows over the picture, so nobody has to keep watching. The frames, the rules and the
model stay in the app's process.

1. Tap **Watch** at the top. A first launch has three rules, one for each example photo: a chessboard
   → a chime; a lighthouse → the line "There is the lighthouse"; a sign with words on it → a count.
2. Choose where the frames come from: **Camera** (the back camera; it asks for the camera permission
   the first time), **Pick a video** (the system picker, no permission needed; the video waits on its
   first frame, Start plays it from the start, muted, and the run ends with the video; a play / pause
   button lets you look through it before), or **Example photos** (the three examples in turn, each on
   the screen for at least 4.5 s).
3. Tap a rule to change it, or **Add a rule** (six at most). The editor has three parts: *Look for*
   (your words; each frame the model is asked "Is there *your words*?"), *Then* (the reaction, below)
   and *Symbol* (one of twelve SF Symbols, on the rule's row and over the picture when it fires; a new
   rule starts with the eye; an effect shows no symbol, so its editor has no *Symbol* part).
   Delete is at the bottom. Rules you edit are kept for the next launch.
4. **Start**. Each row shows the rule's answer for the latest frame (YES, NO, or `?` when the answer
   could not be read) and its reaction; **Stop** ends the run after the frame being answered.

| reaction | when the rule fires | the rule's row shows |
|---|---|---|
| Sound | one of three system sounds: Chime, Glass or Bell | the sound's name |
| Haptic | a short buzz | buzz |
| Say | your line, spoken by the system voice in English, and shown in a bubble over the picture | say |
| Timer | a countdown of the minutes you typed; at its end, the chime and "Time's up" | the time left |
| Count | adds one | ×N |
| Log | the time and the rule as a line under the list (the latest three show) | log |
| Effect | the picture stops on the frame the model answered for 3 s and the app edits it at the box the model gives for the thing, with a sound: Ripple (a wave spreads from the box's middle and leaves the frame tinted blue), Zoom (the frame zooms in on the box, 1.2–2.5×), Spotlight (around the box the frame goes dark and grey); each plays its own sound — running water, a car horn, church bells — unless the editor's Sound row picks another (Water, Horn, Bells, or the system's Chime) | the edit and its sound: ripple · water, zoom · horn, spotlight · bells |

Every rule but an effect also shows its symbol (64 pt) and its words (44 pt) over the picture for at
least 3 s; a spoken line stays until the voice is done. An effect shows no symbol and no words: the
held frame is edited by the app (Core Image; no image) and the row turns YES. The Sound
reaction's sounds are the system's own (`AudioServicesPlaySystemSound`) and follow the ring / silent
switch; the spoken line (`AVSpeechSynthesizer`) and the effects' sounds (`AVAudioPlayer`, the files
`Sounds/water.m4a`, `horn.m4a`, `bells.m4a` in the app — see [Reaction sounds](#reaction-sounds)) play
through it, mixed with other audio. Any effect can play any of the three, or the chime (`-rules` writes
the sound after a `/`: `effect:spotlight/horn`); a build without the file it names plays the chime for it.

### Where an effect plays

An effect plays on the frame the model answered. The moment it fires, the picture shows that frame —
the crop the model got — with a thin blue border for 3 s, and the edit plays on it while its sound plays;
then the live picture comes back. A video is paused for those 3 s and goes on from where it stopped (it is
not rewound); over the camera or an example photo, the held frame covers the live picture. The loop takes
no frame during a hold, and effects that fire on the same frame hold one after the other, in rule order.
The picture moves on while the model answers; the hold shows the frame the answer is about, so the edit
lands where the thing was in that frame.

Where on that frame, the app asks the model (`-cells box`, the default): right after a rule's answer
turns yes, and before the rule fires, a second new session gets the same frame and the model's own
grounding form, "Detect *your words* in the picture and output its bounding box." (at most 48 answer
tokens). The model answers in its own tags — `<ref>your words</ref><box>x1 y1 x2 y2</box>`, or
`The region matching "…" is<box>…</box>.` — left, top, right and bottom in per mille of the frame it got
(0–1000; the kit stretches the frame into the model's square input, so they are per mille of the frame's
width and height). The app reads the first box: the numbers inside the first box tag (`<box>`, or
`<|box_start|>`, `<point>`, `<quad>`), or the first four numbers of an answer without one; four numbers
all within 0–1 with a decimal point count as fractions, and numbers over 1000 as the frame's pixels;
left / right and top / bottom come back in order when the model gives them the other way round. An
answer with fewer than four numbers has no box, and the edit plays on the whole frame.

At the box, Ripple's wave starts at the box's middle, 0.3 of the box's shorter side wide (at least 8 %
of the picture's shorter side); Spotlight keeps the box lit, widened by a tenth of its width and height
on each side, behind a soft edge (a blur of 6 % of the picture's shorter side), and darkens the rest;
Zoom widens the box by a quarter of its width and height on each side, gives that window the picture's
shape and zooms into it in 0.6 s (eased out) — the picture over the window, kept within 1.2–2.5× — moved
in when the window would leave the frame. With no box the edits play as they did before the box: the
wave from the frame's middle, 2× toward the middle, the middle two thirds lit. The edits are Core Image
filters on the held frame (`CIRippleTransition` to a blue-tinted copy; a scaled window; exposure −1.6 EV
and saturation 0.1 outside a blurred mask), one picture per screen frame. The box is the same part of
what the model saw and of what the screen shows: the camera and a video fill the picture's box and the
model gets that crop, so the box's per mille are the picture's; an example photo is fitted whole, so
they are the fitted photo's.

On the example video, a Mac run of the same model with this question, on the 24 frames where a rule
was yes (one frame a second; the thing's box drawn by eye before any answer), put the box's middle on
the river in 8 of 8 frames (the box on the river's band) and on the church in 10 of 11 (a narrow box on
its tower and spire). The model boxes the subject of the sentence: for *a river flowing under a bridge*
it boxed the river — its water, or the whole picture — and the box's middle was on the bridge in 1 of 5
frames; with the bridge as the subject, *a road bridge crossing over the river*, the box sat on the
bridge's deck in 5 of 5, and the sentence was yes on the same seconds (5–9 s). The answers were 32
tokens at most (the median 26; 30 for the bridge sentence).

`-cells ask` asks for one of nine cells instead (3 × 3, numbered 1–9 from the top left, row by row), with
the same second session: with `-cellwords 1` the question is "Where in the picture is *your words*?
Answer with two words: left, center or right; then top, middle or bottom." and the two words make the
cell (a word left out counts as center or middle), without it "Which cell of a 3x3 grid (1-9, numbered
left to right, top to bottom) contains *your words*? Answer with the number only."; the edit then plays
at the cell as at a box. `-cells batch` asks for the cell's number inside the yes / no prompt instead; on
the example video that made the yes / no answers worse. `-cells none` asks nothing: the edit plays on the
whole frame. `-grid 1` draws the nine cells and their numbers on the frame the model gets (the screen
shows them only on a held frame, which is that frame).

On the example video, a Mac run of the same model with the words question named the cell where the
thing was for 79 % of the frames a rule was yes on, and that cell or one next to it for 96 % (24 frames,
three rules) — and answering 5, the middle, every time would have scored 71 % and 96 % on the same
frames: the words beat that fixed guess on 2 of the 24 frames, both where the thing was away from the
middle.

A rule fires when its answer is yes on this frame, was not yes on the frame before, and the rule has
not fired in the last 3 s. A thing that stays in view fires once; one that leaves and comes back fires
again. A count does not count a fire held back by those 3 s.

What one frame sends:

1. The frame: the newest camera frame (1280 × 720 from the back wide-angle camera; the frames that came
   while the model was answering are dropped), the video's frame on the screen at that moment, or the
   example photo on the screen. A camera or video frame is cropped to what the preview shows — the
   preview fills its box, so the model gets the same middle part of the picture; an example photo goes
   whole, as on the Ask screen.
2. A new session (no history, greedy, at most 32 answer tokens) with the frame and one prompt: the line
   "Answer each line with yes or no, nothing else." and a line `<i>. Is there <condition>?` per rule.
3. The answer is read one line per rule: a line that starts with a number answers that rule, its first
   word "yes" or "no" is the answer, and a rule no line answers is `?`.
4. With `-cells box` (the default), for each effect rule whose answer turned yes: a second new session
   with the same frame and the box question (at most 48 answer tokens), then the rule fires at that box.
   With `-cells ask`, for each rule whose answer turned yes: the same with the cell question (at most 8
   answer tokens), then the rule fires at that cell.
5. The next frame is taken when the answers are in and the holds of the effects that fired are over. At
   the thermal state fair a frame starts at most every 2 s, at serious every 4 s, and the second numbers
   line says so (`every 2.0 s`).

The numbers lines under Start (as the screen showed them at the 18th frame of the Example photos run below):

```
first token 1.27 s · total 1.37 s
GPU · on-device · frame 18 · nominal · 349 MB
```

- first token: from the frame's call to the model to its first streamed text.
- total: from taking the frame to its answer read — one turn of the loop. The place question (the box,
  or the cell of `-cells ask`) and an effect's hold come after it and are not in these numbers (the record
  keeps them: `place_ms`, `cell_ms`, `hold_ms`).
- GPU · on-device: where the model ran, as on the Ask screen; then the frames answered since Start, the
  thermal state (`ProcessInfo.thermalState`) and the app's memory footprint (`phys_footprint`).

On an iPhone 18 Pro (iOS 27.2) with the three first-launch rules, every frame sent 123 tokens in and got
11 back, one numbered line per rule (`1. No`, `2. No`, `3. Yes`). With Example photos, a 40 s run answered
30 frames, a frame took 1,366 ms (median; the first token came after 1,276 ms), all 90 answers matched
the photos, and the rules fired 8 times — the chime 3 times, the spoken line twice, the count 3 times —
each 1.36–1.40 s after its photo came on the screen. With the camera facing a dark desk, a 30 s run
answered 21 frames, a frame took 1,455 ms (median; the frame was 31 ms old when it reached the model),
all 63 answers were no, and nothing fired. Both runs stayed at the thermal state nominal, and the memory
footprint stayed between 335 and 383 MB. Measured on 2026-10-21.

Take 5 ran the build before the box (Look 1.0 (1) built from commit 4346afa: the place asked in words, the
edit at a cell, each edit with its own sound): the
[example video](#example-video) with three effect rules — *water rushing over rows of stones* → Ripple,
*a river flowing under a bridge* → Zoom, *a church with a tall spire* → Spotlight — and the place question
in words (`-cells ask -cellwords 1`), in a new process on the same phone, on 2026-10-23
(`watch-run-1792743006.json`). It answered 15 frames and ended with the video after 35.2 s: the 26 s of the
video and its three holds, the video standing still during each. A frame took 1,498 ms (median; the first
token came after 1,380 ms), with 130 tokens in and 11–14 back, and each place question took 1,213–1,224 ms
more (median 1,213 ms). At the end the numbers lines read `first token 1.37 s · total 1.49 s` and
`GPU · on-device · frame 15 · nominal · 336 MB` (the last frame's). Each rule fired once and nothing else
did, on the frames taken 0.02, 5.70 and 15.85 s into the video — within a second of the seconds where the
same model with the same prompt, given one frame a second on a Mac, first answers yes (0, 5 and 15 s):

| rule | the place answer | cell | where the thing was, by eye (the seconds around that frame) | the edit on the held frame |
|---|---|---|---|---|
| water rushing over rows of stones | `center; middle` | 5 | cells 2, 4, 5, 7, 8 | Ripple from the middle of cell 5 |
| a river flowing under a bridge | `center; top` | 2 | cell 3, the next cell | Zoom to 2× toward the middle of cell 2: the bridge's abutment at the window's top right |
| a church with a tall spire | `center; top` | 2 | cells 2, 3 | Spotlight on cell 2: the church tower |

Each hold started 2.68–2.72 s after its frame was taken (the yes / no answer, then the place question), with
its edit and, 19–41 ms later, its sound; it held the frame for 3,006–3,029 ms, and the video went on from the
second where it had stopped. The thermal state stayed nominal and the memory footprint between 336 and
410 MB.

Take 4, below, ran the build before the hold (commit 6aa2700), which drew its effects — Ripple, Sweep and
Rays, animations of its own — over the live picture instead of editing a held frame. That take played the
[example video](#example-video) with three effect rules — *water rushing over rows of stones* → Ripple,
*a river flowing under a bridge* → Sweep, *a church with a tall spire* → Rays — and the place question in
words (`-cells ask -cellwords 1`), in a new process on the same phone,
on 2026-10-22 (`watch-run-1792648309.json`; Look 1.0 (1) built from commit 6aa2700). It answered 15
frames in 26.2 s and ended with the video; a frame took 1,495 ms (median; the first token came after
1,382 ms), with 130 tokens in and 11–14 back, and each place question took 1,213–1,228 ms more (median
1,223 ms). At the end the numbers lines read `first token 1.37 s · total 1.49 s` and
`GPU · on-device · frame 15 · nominal · 349 MB` (the last frame's). Each rule fired once and nothing else
did, on the frames taken 0.03, 5.74 and 15.94 s into the video — within a second of the seconds where the
same model with the same prompt, given one frame a second on a Mac, first answers yes (0, 5 and 15 s):

| rule | the place answer | cell | where the thing was, by eye (the seconds around that frame) |
|---|---|---|---|
| water rushing over rows of stones | `center; middle` | 5 | cells 2, 4, 5, 7, 8 |
| a river flowing under a bridge | `center; top` | 2 | cell 3, the next cell |
| a church with a tall spire | `center; top` | 2 | cells 2, 3 |

Each effect started 2.68–2.72 s after its frame was taken (the yes / no answer, then the place question),
and its sound 16–19 ms after that; the video moved on meanwhile, so an effect marked where the thing had
been when its frame was taken — what the hold now shows. The thermal state stayed nominal and the memory
footprint between 349 and 414 MB. An earlier take of the same video with three plain reactions (the chime,
a count and a spoken line, no place question; `watch-run-1792603139.json`) answered 20 frames at 1,345 ms a
frame (median), and each reaction came 1.34–1.37 s after its frame was taken: the next frame is taken when
the answer before it is in, so a thing that comes into view waits for that frame.

The screen sits in the same band as the Ask screen. On an iPhone 18 Pro with three rules the picture is
370 × 275 pt with `-take 1` (the run's record, `preview_pt`) and ≈ 229 pt tall with Add a rule under
the rules (by the rows' heights); with more than three rules the list scrolls (a take shows the first
three).

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
| `-take 1` | hide the Ask \| Watch switch, so the band starts right under the status bar (kept in the record as `take`) |

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

### The watch screen

`-watch 1` opens the app on the Watch screen. With `-seconds`, the app also runs a watch by itself: it
waits for READY (and the camera, or the `-file` video), then for `-trigger`, then presses Start, watches
for `-seconds` (a run over a video ends sooner when the video does) and presses Stop — `start()` and
`stop()`, the functions the Start / Stop button calls. Without `-seconds` it presses nothing and the
screen waits for a finger.

```sh
xcrun devicectl device process launch --device <udid> com.coreaikit.look -- \
  -watch 1 -source photos -seconds 40 -take 1 -frames 1 -log 1 \
  -rules "a chessboard=chime|a lighthouse=say:There is the lighthouse|a sign with words on it=count"
```

| argument | meaning |
|---|---|
| `-watch 1` | open on the Watch screen |
| `-source camera` | `camera` (the default on an iPhone), `video` (with `-file`) or `photos` |
| `-rules "a=b\|…"` | this launch's rules (not kept): `condition=reaction`, the reaction one of `chime`, `glass`, `bell`, `haptic`, `say:<line>`, `timer:<minutes>`, `count`, `log`, `effect:ripple`, `effect:zoom`, `effect:spotlight` — with another sound than the edit's own after a `/`: `effect:spotlight/horn` (`water`, `horn`, `bells` or `chime`) |
| `-cells box` | where an effect's edit plays on the held frame: `box` (the default: a second question about the frame for each effect rule that turns yes, before it fires — the model's box), `ask` (a cell: a second question for each rule that turns yes), `batch` (the cell's number inside the yes / no prompt) or `none` (not asked: the whole frame) |
| `-cellwords 1` | with `-cells ask`: the place as two words (left / center / right, top / middle / bottom) instead of a cell's number |
| `-grid 1` | draw the nine cells and their numbers on the frame the model gets (the screen shows them only on a held frame) |
| `-conditions "a\|b\|…"` | the first version's form: conditions only, each with the chime |
| `-symbols "a\|b\|…"` | the rules' SF Symbols, by place |
| `-seconds 40` | press Start, watch this long (a frame being answered is finished), press Stop |
| `-trigger <file>` | wait until this file exists in Documents before Start |
| `-delay 1` | seconds between READY (or the trigger) and Start |
| `-file <path>` | a video file opened as the video source, as a picked video is: `Documents/<name>` in the app's container (where `devicectl device copy to` puts it), a name in Documents, or an absolute path (`-video <name>` does the same) |
| `-hold 4.5` | seconds each example photo stays on the screen; `0` = the next photo every frame |
| `-tokens 32` | the answer's token cap |
| `-truth "sign:no,no,yes;…"` | what each example photo shows, rule by rule (built in when every condition is one the app knows: a chessboard, a lighthouse, a sign with words on it, a person) |
| `-permwait 600` | how long the first camera run waits for the answer to the camera permission request |
| `-edit new`, `-edit <n>` | open the rule editor at READY, for a new rule or for rule n |
| `-add "cond=reaction"` | with `-edit new`: fill the new rule, then save it `-gap` seconds later (what Done calls); alone: add it at READY |
| `-take 1` | hide the Ask \| Watch switch; at most three rules show, and the video's play / pause button is hidden |
| `-frames 1` | write the screen as PNGs to `Documents/frames/watch-<epoch>/`: READY, Start, each reaction (0.3 s after it) and the end; and for each effect its edited held frame 2.5 s into the hold (`catch<n>.png`, n = the fire's number) |
| `-log 1` | the lines and the record below, for every run (Start to Stop), by hand or not |
| `-quit 1` | exit after the record; without it the app stays on the screen as the run left it |

The log (`Documents/look-autoplay.log`) has START (with `cells=`, `grid=`, `sounds=` — the effect
sounds this build has — and `record=`, the record's schema) and READY, CAMERA / VIDEO when the source
opens, TRIGGER, PRESS start / PRESS stop (VIDEO ended when the video ends the run), one WATCH line per
frame (its times, tokens, answers, the rules that fired; `cells=` and `cell_ms=` when a cell is asked,
`place_ms=` when a box is), one FIRE line per reaction (the rule, the reaction, the frame, the time,
`cell=`, `effect=`, `sound=` — the sound's id, or `fallback` for the chime — `edit=` and `hold_ms=`, how
long it holds the frame: 3000 for an effect, 0 otherwise, then `box=` the box its edit plays at,
`x1,y1,x2,y2` or `-` for none, and `place_ms=` the box question's time), one HOLD line when an effect's
hold ends (`rule=`, `frame=`, `held_ms=` its length, `resumed_t=` where the video went on, in seconds),
TIMER when a timer ends, EDIT and RULES for the editor, and DONE (or ERROR) with the record's name. An
effect's FIRE line comes when its hold starts, after its frame's WATCH line, and its HOLD line before the
next WATCH line. The record (`Documents/watch-run-<epoch>.json`, schema `watch-run/5`) holds the rules (an
effect's with the `sound` it plays) and the prompt, `cells_form`, `cell_words`, `cell_prompt`,
`place_prompt` and `place_max_tokens` (the box question, `<condition>` for the rule's words), and `grid`
(and `grid_style`), and `sounds_in_build`; per frame the times (capture, grab, done), the crop, the tokens
in and out, the answers and the raw text, `cells` (the cell each rule's answer named, null for none),
`cell_ms` and `cell_asks` (each cell question: the rule, the raw answer, the cell, its times), `boxes`
(the box each rule got, null for none), `place_ms` and `box_asks` (each box question: the rule, the raw
answer's first 200 characters, the box, its times and tokens), the rules fired and held back, `hold_ms`
(the holds that followed the frame, 0 for none), the thermal state and the memory footprint, and with
Example photos what the photo shows and when it came on the screen; `fires` (each reaction: the rule, the
frame, the time from the frame's capture — and with Example photos from the photo's change — to the
reaction (for an effect, to its hold's start), `cell` and `effect`; for an effect `edit`, `box` (four
integers 0–1000, null for none), `box_unit` (the box on the picture: x, y, width, height in units of the
picture), `place_text` (the box answer's first 200 characters) and `place_ms`, `held_frame` (the frame it
holds, the one that fired), `hold_ms` (the hold's length, measured), `held_png` (with `-frames 1`), and for
a video `paused_t_s` / `resumed_t_s` (where the video stopped and went on), `sound` (the id it played, or
`fallback` when this build has not its file) and `sound_chosen` (the rule's), `sound_ms` until the player
started (until the system reported the chime played: `sound_done_ms`), `sound_s` its length and
`effect_rect_pt` the place on the picture; `sound_ms` for the Sound reaction until the system reported it
played; `speech_start_ms` / `speech_end_ms` for a spoken line), `suppressed` (a yes within 3 s of the
rule's last fire), `timers`, `presses`; the band in pixels (`take_rect_px`), `screen_px`, the picture's size
(`preview_pt`), the numbers lines, the thermal state and the free bytes at the start and the end, the
stretches when the app was not active (`inactive_*`), and `frames_png`. The summary holds the frame count,
the median and range of the times (`cell_ms` and `place_ms` too, and the box answers' tokens:
`place_tokens`), the yes count per rule, the unread answers, the fires by rule, by reaction, by effect, by
edit and by sound, `holds` (how many, their total, shortest and longest ms), the cells and the boxes named
per rule (`cells_named`, `boxes_named`), the fires with a cell and with a box, and the thermal states seen.

The first version of this mode (`-conditions` with four conditions — a chessboard, a lighthouse, a sign
with words on it, a person — and the next example photo every frame, as `-hold 0` does now) was
measured on an iPhone 18 Pro (iOS 27.2) on 2026-10-21: every frame sent 131 tokens in and got 15 back,
one numbered line per condition (`1. No`, `2. No`, `3. Yes`, `4. No`). With the example photos the loop
answered 41 frames in 61 s, a frame took 1,485 ms (median; the first token came after 1,355 ms), and
all 164 answers matched the photos. With the camera facing a dark desk it answered 39 frames in 62 s,
a frame took 1,576 ms (median; the frame was 32 ms old when it reached the model, and the first token
came 1,443 ms later), and all 156 answers were no. The memory footprint stayed between 334 and 353 MB.
Both runs started at the thermal state nominal and ended at fair (the photos run in its last frame, the
camera run from 47 s on).

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

## Example video

One video for the Watch screen: [Drone video of Kose fish ladder and church](https://commons.wikimedia.org/wiki/File:Drone_video_of_Kose_fish_ladder_and_church.webm)
on Wikimedia Commons, CC0 1.0 according to its file page (the license template and the file's metadata).
A drone looks straight down at a river with stones, a road bridge crosses into view, then the drone
climbs and turns to a white church with a tall spire among houses and trees. No face, text, brand or
logo can be read in it.

It is not in the repository, and the app does not bundle it. `tools/fetch_examples.sh` downloads the
original (WebM, 2720 × 1530, 35 s, 283,686,946 bytes) into `Videos/src/`, checks its size and sha256,
and cuts the clip the app plays with ffmpeg (`brew install ffmpeg`): 26 s from 3.0 s of the original,
1920 × 1080, 30 fps, H.264, no audio, into `Videos/flyover.mp4`. The same ffmpeg (9.0.1) makes the same
bytes (`cut_sha256` in `tools/examples.json`); with another version the bytes can differ, and the script
checks the clip's form instead (H.264, 1920 × 1080, 30 fps, 26 s, no audio). Without ffmpeg the script
says so and puts the photos in place only.

To watch it on the phone: copy `Videos/flyover.mp4` to the iPhone (AirDrop from a Mac puts it in
Photos), open Watch, tap **Pick a video** and choose it, write three rules as states and relations —
*water rushing over rows of stones* → Effect (Ripple), *a road bridge crossing over the river* → Effect
(Spotlight, Sound Horn), *a church with a tall spire* → Effect (Zoom, Sound Bells) — and Start. The video
plays once from its start, muted, pauses for 3 s at each effect, and the run ends with it; each edit plays
at the box the model gives for its rule's words, and the effects' sounds come from
`tools/fetch_examples.sh`. A script starts the same run with launch arguments (a recorded take adds
`-trigger` and `-frames 1`):

```sh
xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
  --domain-identifier com.coreaikit.look --source Videos/flyover.mp4 --destination Documents/flyover.mp4
xcrun devicectl device process launch --device <udid> com.coreaikit.look -- \
  -watch 1 -source video -file Documents/flyover.mp4 -seconds 45 -delay 2 -take 1 -log 1 \
  -cells box \
  -rules "water rushing over rows of stones=effect:ripple|a road bridge crossing over the river=effect:spotlight/horn|a church with a tall spire=effect:zoom/bells" \
  -symbols "water.waves|road.lanes|building.columns"
```

Take 5's run asked for a cell in words instead (`-cells ask -cellwords 1`), with *a river flowing under a
bridge* → Zoom and *a church with a tall spire* → Spotlight, each edit with its own sound.

The earlier take of the same video had three plain reactions — *a river* → Sound (Chime), *a bridge* →
Count, *a church* → Say "There is a church":

```sh
xcrun devicectl device process launch --device <udid> com.coreaikit.look -- \
  -watch 1 -source video -file Documents/flyover.mp4 -seconds 45 -delay 2 -take 1 -log 1 \
  -rules "a river=chime|a bridge=count|a church=say:There is a church" \
  -symbols "water.waves|road.lanes|building.columns"
```

## Reaction sounds

The three sounds the effects play, each CC0 1.0 according to its Wikimedia Commons file page (the license
template and the file's metadata); none has a voice or music in it. They are not in the repository:
`tools/fetch_examples.sh` downloads each original into `SoundSources/<id>/`, checks its size and sha256,
and cuts the clip the app plays with ffmpeg into `Sounds/`, the folder the app bundles: mixed to mono,
48 kHz, the span and the fades the manifest names, the gain that brings it to −20 LUFS, AAC 96 kbit/s.
The same ffmpeg (9.0.1) makes the same bytes (`cut_sha256` in `tools/examples.json`); with another
version the script checks the clip's form, length and loudness instead. Without ffmpeg the script says
so, and the app plays the system chime for each effect. Each edit plays its own sound unless its rule
picks another of the three, or the chime (the editor's Sound row; `effect:<edit>/<sound>` in `-rules`).

| id | the edit it is the own sound of | sounds like | source page |
|---|---|---|---|
| `water` | Ripple | a shallow brook over stones, 2.3 s | [Commons](https://commons.wikimedia.org/wiki/File:2024-07-26_Molln_(Ober%C3%B6sterreich)_Bachlauf_pl%C3%A4tschert_(Krumme_Steyerling_bei_Piesslingerstra%C3%9Fe).wav) |
| `horn` | Zoom | a car horn pressed twice, 2.3 s | [Commons](https://commons.wikimedia.org/wiki/File:Car_Horn.wav) |
| `bells` | Spotlight | one church bell swinging, four strikes, 3.0 s | [Commons](https://commons.wikimedia.org/wiki/File:Zajeci_zvony_kostel_2025.ogg) |

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
| Watch: the camera permission refused (or turned off later) | Camera access is off: Settings › Privacy › Camera |
| Watch: the first camera use, before the answer to the permission request | Allow the camera to start watching |
| Watch: Pick a video before a video is picked | Pick a video, then Start |
| Watch: a picked video the app cannot open | Could not read that video / Could not open that video, and the reason |
| Watch: every rule deleted | Add a rule first (Start stays off) |
| Watch: Stop during a frame | Stopping after this frame…, then PAUSED · Stopped · N frames · M reactions |
| Watch: an effect whose sound file is not in the build | nothing: the effect plays with the system chime (the record says `fallback`) |
| Watch: the box answer has fewer than four numbers in it | nothing: the edit plays on the whole held frame (the record's `box` is null) |
| Watch: the cell answer (`-cells ask`) has no number or place word in it | nothing: the edit plays on the whole held frame (the record's `cell` is null) |
| Watch: Stop while an effect holds the frame | Stopping after this frame…; the hold plays its 3 s first, then PAUSED |

These are the app's own sentences, from `LookModel.swift` and `Watch.swift`.

## Models

`-model <catalog id>` loads another `vlm` row of the catalog the same way (`KitVisionModel(catalog:)`):
`qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` (Mac only), `holo2-4b`, `minicpm-v-4.6`, `lfm2.5-vl-450m`,
`lfm2.5-vl-3b`, `north-micro-vision`. Look has been run with `minicpm-v-4.7-1b` only; the others are
untested in this app. The numbers line's backend and the footer come from the row that loaded.
