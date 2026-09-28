# Week — a calendar week planned on device

One tap, and every event of the week gets what it needs before it: nothing, documents, preparing
something, leaving early, a link to join, something to buy or bring, or a reply by a deadline. The
events that need something become reminders. The model reads each event's title, place and notes on
the device; nothing leaves it.

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

## Run

```bash
swift run -c release week-cli run --count 20 --seed 7 --out results/mac-week-20.json
swift run -c release week-cli run --bundle <dir> --events week.json --out result.json
swift run week-cli dump --count 20 --seed 7 --out week.json     # the events only, no model
swift test                                                      # the generator's contract
```

Without `--bundle` the CLI downloads the catalog's `decider-0.8b` (1.34 GB) on use and caches it.
`--bundle` takes a local export (`metadata.json`, the `.aimodel`, `tokenizer/`). The model loads,
one throwaway decision warms it, then the clock starts. Progress and every answer go to stderr, one
summary JSON line to stdout, and `--out` gets the whole result: every event's answer with all seven
probabilities, the per-event series `[seconds after the start, ms, prompt tokens]`, the medians and
where it ran.

## The app

`App/` is the same planner as a screen, on an iPhone or a Mac:

```bash
cd App && xcodegen generate && open Week.xcodeproj   # export DEVELOPMENT_TEAM=… before, or pick the team in Xcode
```

1. On launch it asks for full access to Calendar, then Reminders. It keeps its own calendar,
   **Demo week**, in the device's local source, writes the synthetic week into it when it holds no
   event this week, and reads that calendar alone.
2. A spotlight card above the list shows one event large: at READY the first event with the question and
   the seven options, while planning the event whose answer arrived last, with its chip.
3. **Plan my week** asks the model about every event in time order. Each row gets a chip as its
   answer arrives, the seven answers grow as bars, and **Before your week** lists the events that
   need something. The clock, the median and the rate on screen are the app's own measurements.
4. **Add N reminders** puts one reminder per listed event into a **Before your week** list: the
   title is `<what>: <event>`, it is due the day before the event at 9:00, and its notes are the
   event's. Pressing it again adds nothing twice.

`-store 0` keeps everything in the app: the generator's week, no calendar read, no reminders
written, and the DONE line says `in-app sample week`.

The calendar and the list are written only into the local source ("On My iPhone"). With no
local source the app plans the in-app week and says `calendar_source: none (no local source;
in-app sample week)` on the READY line, in `access.json` and in the result file; a local source
that refuses them gives the in-app week too, with the reason in `calendar_source`. A synced
source (iCloud, CalDAV, Exchange) would carry both to every device on the account, so the app
writes into the default source only when launched with `-syncedStore 1`.

The model comes from the catalog on launch (downloaded once, cached), or from a folder already on
the device: on an iPhone a copy of the bundle at `Documents/decider-0.8b/` (put there with
`xcrun devicectl device copy to`), on a Mac `-bundle <dir>`.

The week is generated, not collected (`Sources/WeekCore/Week.swift`): invented events with given
names only, generic places, `555` phone numbers and `example` links; every proper name it can
write is in `NAMES.txt`. A seed gives the same week on every platform.

For a recording or an unattended run, the app presses its buttons itself:

```
Week -autoplay 1 -delay 3 -count 20 -seed 7 -log 1 [-trigger <file>] [-reminders 1] [-store 0] [-bundle <dir>]
```

With `-log 1` the status line goes to `Documents/week-autoplay.log` as it changes, and every
finished run writes `Documents/week-result-<epoch>.json` (the CLI's fields, plus the calendar
source and `reminders_added`, rewritten when `-reminders 1` adds them), which
`devicectl device copy from` reads back from a phone (put `--` before the app's arguments in
`devicectl device process launch`, or it reads `-log` as its own option).

A phone nobody taps still has to answer the two permission alerts once. `WeekUITests` does it: it
launches the app with `-grantOnly 1` (ask, show the answer, write `Documents/access.json`, load no
model), taps the alerts' "Allow" buttons and checks the app's answer.

```bash
xcodebuild test -project App/Week.xcodeproj -scheme Week -destination "id=<device udid>" \
    -only-testing:WeekUITests/WeekUITests/testGrantAccess
```

It does not run unattended. On an iPhone 18 Pro (iOS 27.0), before the test could start, iOS asked
on the phone for its passcode ("Enter iPhone Passcode for “XCTest”", to enable UI Automation). With
nobody at the phone, `xcodebuild` stopped after a minute with "Timed out while enabling automation
mode", and the app never asked for access. A person has to type the passcode on the phone before this
test can run; without that, run the app with `-store 0`.

## Measured

2026-09-27, seed 7, 20 events, `week-cli run` on an M4 Max (Mac16,9, macOS 27.0, GPU, the int8
bundle): 12.3 s in all, 619 ms median and 652 ms p90 per event, 97 to 122 prompt tokens an event
(`results/mac-week-20.json`). The app on the same Mac (`-store 0`, its own clock) gave the same
answers in 12.5 to 13.2 s over four runs, 621 to 659 ms median. 19 of the 20 answers are the
kind the generator wrote the event as; the other is a potluck where you bring the salad and
plates, answered as preparing something.

On an **iPhone 18 Pro** (iPhone19,2, iOS 27.0, 2026-09-28, the same int8 bundle copied into
`Documents/decider-0.8b`, `-store 0`, the phone at thermal state nominal from start to finish, the
app's own clock): the recorded run planned the 20 events in 22.2 s, 1,119 ms median and 1,182 ms
p90 per event, 0.90 events/s, the same 20 answers as the Mac; three earlier runs gave 22.0 s and
1,105 to 1,108 ms median (`results/iphone/week-result-*.json`). The model loads in 1.5 s once its
compiled form is cached (4.6 s on the first launch). Per prompt token the phone takes about 10 ms
against the Mac's 5.6 ms. The recorded clip (2026-09-28, `-count 14`, the same phone and bundle) planned
14 events in 15.5 s, 1,088 ms median per event, the spotlight card showing each event and its answer as
it lands (`results/iphone/week-result-1790556845-recorded.json`).
