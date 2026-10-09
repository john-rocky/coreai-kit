# Week — your week, planned on device

`decider-0.8b` (`mlboydaisuke/decider-0.8b-CoreAI`) on Core AI reads each event of your week and says
what it needs before it: 627 to 651 ms per event (median, two runs) on an M4 Max, measured 2026-10-09.

Week takes the week from your calendar, from lines you paste, or from a sample week. One tap asks the
model one question per event. The events that need something are listed under **Before your week**,
in time order. Nothing is written to your calendar, and nothing leaves the device.

The model is [decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) (Mapika, Apache-2.0)
exported to Core AI (the catalog's `decider-0.8b`), and each event is one typed decision:

```swift
let decider = try await TypedDecisions(catalog: "decider-0.8b")   // downloaded once, cached
let answer = try await decider.decide(
    "Mon 11:00 Lease renewal signing · the property office, 12 Weaver St · Notes: bring two forms of ID",
    .choice("What does this calendar event need you to do before it?", [
        "nothing to prepare", "bring a document or an ID", "prepare or send something first",
        "travel time: leave early", "join online: a link or dial-in", "buy or bring something",
        "confirm or reply by a deadline",
    ]))
answer.choice       // "bring a document or an ID"
answer.confidence   // its probability
```

Two choices shape the loop (`Sources/WeekCore/WeekPlanner.swift`):

- **One event, one prompt.** Each event is its own state. Asked about "event 7" of a whole week in
  one state, the 0.8B model loses track of which event is meant (8 of 20 right, against 18 of 20 one
  event at a time, in a 20-event panel on a Mac).
- **Short options.** Every option is part of every prompt, and this bundle's graph takes one token
  per step, so a prompt costs its length: about 110 tokens an event with these seven.

## Where the week comes from

Three buttons at the top pick the source.

- **Your calendar**, the default. Week asks for full Calendar access, then reads this week, Monday to
  Sunday, from every calendar on the device. All-day, cancelled and declined events are left out.
- **Paste**. One event a line, in the form the model reads:

  ```
  Mon 11:00 Lease renewal signing · the property office · Notes: bring two forms of ID
  ```

  The day is `Mon` or `Monday`, the time `H:MM` or `HH:MM` (24-hour). The place and `Notes:` follow
  ` · ` and are optional. A line without a day, a time and a title is skipped, and the sheet counts
  it ("2 lines skipped"). A week holds at most 200 events. **Import .json** fills the sheet from a
  `week-cli dump --out` file.
- **Sample week**. The generator's 20 events (seed 7), the week `week-cli run` plans. It is
  generated, not collected (`Sources/WeekCore/Week.swift`): given names only, generic places, `555`
  phone numbers and `example` links. Every proper name it can write is in `NAMES.txt`.

The model reads an event's title, place and notes as one line. Each is cut at 120, 120 and 240
characters, because every character costs prompt time; a meeting invite puts its link near the top.

With nothing to plan, the screen says why and offers Paste and the sample week. The reasons:
calendar access is off, no events this week, nothing pasted, no readable line, or more than 200 events.

## The screen

1. A spotlight card shows one event large. Before a run it is the week's earliest event, with the question
   and the seven options; while planning, the event whose answer arrived last, with its chip.
2. **Plan my week** asks the model about every event in time order. Each row gets a chip as its
   answer arrives, and the seven answers grow as bars. The clock, the median and the rate on screen
   are the app's own measurements.
3. **Add N reminders** appears after a run on your calendar or a paste, when an event that needs
   something is still ahead. Only then does Week ask for Reminders access. Each such event gets one
   reminder in your default Reminders list: titled `<what>: <event>`, due the day before at 9:00,
   with the event's notes. Pressing again adds nothing twice. The sample week gets no reminders.

## Measured

2026-10-09, an M4 Max (Mac16,9), macOS 27.0 (26A428), a Release build on the GPU, the catalog's int8
bundle at revision `ff60ccf`. Each run is a fresh launch: load, one warm-up decision, then the press.
All four ran in one 87-second measurement window, opened after the GPU had sat at 5 % or less for 60 s.
macOS's photo analysis (`mediaanalysisd`) used one to two CPU cores at the window's start and end.

| Run | Events | Total | Median per event | p90 | Load | Warm-up |
|---|---|---|---|---|---|---|
| App, the sample week pasted, run 1 | 20 | 12.90 s | 627 ms | 668 ms | 1.00 s | 0.60 s |
| App, the same paste, run 2 | 20 | 13.01 s | 651 ms | 695 ms | 0.99 s | 0.62 s |
| `week-cli run --events`, the same 20 | 20 | 12.91 s | 658 ms | 677 ms | 1.61 s | 0.68 s |
| App, this Mac's own calendar | 7 | 4.59 s | 640 ms | 691 ms | 0.70 s | 0.65 s |

Load is from the cache, with the compiled form already cached by earlier launches. Without the cache, a
launch also downloads the bundle and compiles it; that was not timed in the window.

In both app runs the 20 answers equal `week-cli`'s, event by event: 4 nothing to prepare, 3 documents,
4 prepare, 3 leave early, 3 join online, 1 buy or bring, 2 confirm. 19 of the 20 are the kind the
generator wrote the event as. The other is a potluck where you bring the salad and plates, answered as
preparing something. A prompt is 97 to 122 tokens.

On an iPhone 18 Pro (iPhone19,2, iOS 27.0, 2026-09-28), the app before the calendar and paste inputs
planned the same 20 events in 22.2 s, 1,119 ms median per event (the sample week kept in the app, the
int8 bundle copied to the phone). This version was built for iOS, not run on a phone.

## Run

```bash
cd App && open Week.xcodeproj        # pick your team in Signing & Capabilities, then Run
cd App && xcodegen generate          # after editing project.yml
xcodebuild -project App/Week.xcodeproj -scheme Week -destination 'platform=macOS' DEVELOPMENT_TEAM=<team> build
```

The team goes to `xcodebuild` as a build setting: the project holds `${DEVELOPMENT_TEAM}`, which an
exported environment variable does not fill at build time.

```bash
swift run -c release week-cli run --events week.json --out result.json   # a week from a file
swift run -c release week-cli run --count 20 --seed 7                     # the sample week
swift run week-cli dump --count 20 --seed 7 --out week.json   # stdout: the 20 lines, in the Paste form
swift test                                                    # the generator's and the parser's contract
```

Without `--bundle` the CLI downloads the catalog's `decider-0.8b` (1.34 GB) on use and caches it.
`--bundle` takes a local export (`metadata.json`, the `.aimodel`, `tokenizer/`). The model loads,
one throwaway decision warms it, then the clock starts. Progress and every answer go to stderr, one
summary JSON line to stdout, and `--out` gets the whole result: every event's answer with all seven
probabilities, the per-event series `[seconds after the start, ms, prompt tokens]`, the medians and
where it ran.

The model comes from the catalog on launch (downloaded once, cached), or from a folder already on
the device: on an iPhone a copy of the bundle at `Documents/decider-0.8b/` (put there with
`xcrun devicectl device copy to`), on a Mac `-bundle <dir>`.

## Unattended runs

For a recording or a scripted check, the app presses its own buttons:

```
Week -autoplay 1 -source calendar|paste|sample [-events <.txt or .json>] -delay 3 -log 1 \
     [-out <dir>] [-trigger <file>] [-reminders 1] [-bundle <dir>]
```

`-source paste -events <file>` opens the Paste sheet on the file and presses the sheet's Plan, so a
file gets the reading a paste gets, refusals included. With `-log 1` the status line goes to
`week-autoplay.log` as it changes, and every finished run writes `week-result-<epoch>.json`: the
CLI's fields plus `source` (calendar, paste or sample) and `skipped_lines`. For your calendar's week
the file keeps the answers and the times, and each event is `calendar event <n>`, never its text.
Both files go to `-out`, else Documents. On a Mac pass `-out`: Documents asks for folder access and
may sync to iCloud. On an iPhone, `devicectl device copy from` reads Documents back (put `--` before
the app's arguments in `devicectl device process launch`, or it reads `-log` as its own option).

A phone nobody taps still has to answer the two permission alerts once. `WeekUITests` does it: it
launches the app with `-grantOnly 1` (ask for both, show the answer, write `access.json`, load no
model), taps the alerts' "Allow" buttons and checks the app's answer.

```bash
xcodebuild test -project App/Week.xcodeproj -scheme Week -destination "id=<device udid>" \
    -only-testing:WeekUITests/WeekUITests/testGrantAccess
```

It does not run unattended. On an iPhone 18 Pro (iOS 27.0), before the test could start, iOS asked
on the phone for its passcode ("Enter iPhone Passcode for “XCTest”", to enable UI Automation). With
nobody at the phone, `xcodebuild` stopped after a minute with "Timed out while enabling automation
mode", and the app never asked for access. A person has to type the passcode on the phone before
this test can run; without that, a person answers the alerts in the app, or uses Paste or the sample
week, which ask for nothing.
