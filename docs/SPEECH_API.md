# Design — the speech API

For an app engineer who wants speech in their app and has never trained a model. The test for
every decision here: can someone picture the code after reading one example.

Speech is the largest demand on the map by roughly 2× (see [`TASK_MAP.md`](TASK_MAP.md)), and it
contains the one capability Apple ships no answer for — who spoke when. What is missing today is
not models. It is that **nothing is live**: `MicRecorder` records, you stop, then you transcribe.

---

## Input: four shapes, each obviously itself

```swift
// 1. a file
let text = try await CoreAI.transcribe(voiceMemoURL)

// 2. samples you already have — [Float], 16 kHz mono
let text = try await CoreAI.transcribe(samples)

// 3. a long file, with results as they arrive
for try await u in CoreAI.transcribeStream(lectureURL) { … }

// 4. the microphone
for try await u in CoreAI.listen() { … }
```

1 and 2 exist. 3 and 4 are the work.

`listen()` owns the things an engineer should not have to: microphone permission, the audio
session, interruptions (a phone call arrives), and route changes (headphones unplugged). If the
adopter has to handle `AVAudioSession` themselves, the job was not removed.

## One element type, streamed

```swift
public struct Utterance: Sendable, Hashable {
    public let text: String
    public let speaker: Int?                       // nil until diarization resolves it
    public let range: ClosedRange<TimeInterval>
    public let isFinal: Bool                       // false: this text will still change
}
```

`MeetingTurn` becomes a deprecated alias rather than a second type living beside it. Two shapes
for the same idea is how the two halves drift — the pattern already cost this project a
contributor's file and a set of catalog pins in one day.

The batch API keeps returning `MeetingTranscript`; its `turns` are simply the finalized
utterances.

## What it looks like in a view

```swift
@State private var confirmed = ""
@State private var live = ""

var body: some View {
    VStack(alignment: .leading) {
        Text(confirmed)
        Text(live).foregroundStyle(.secondary)      // not final yet, so show it as provisional
    }
    .task {
        for try await u in CoreAI.listen() {
            if u.isFinal { confirmed += u.text } else { live = u.text }
        }
    }
}
```

Two labels and a loop. Speakers are one argument away:

```swift
for try await u in CoreAI.listen(diarize: true) {
    // u.speaker is 0, 1, 2 … once it is known
}
```

## Speaking

```swift
try await CoreAI.say("Ready.")                  // plays it
let audio = try await CoreAI.speak(text)        // gives you the samples (exists)
```

An engineer adding a spoken response wants sound to come out, not a buffer to manage. `say` is
the verb; `speak` stays for anyone routing audio themselves.

## The whole loop

```swift
for try await u in CoreAI.listen() where u.isFinal {
    let reply = try await CoreAI.chat(u.text)
    try await CoreAI.say(reply)
}
```

A voice interface with no server, no API key, and no per-minute cost. This is the example to
lead with, because it is the one that is impossible any other way.

---

## How the streaming actually works

Worth stating plainly, because it explains why the API has the shape it does.

1. `AVAudioEngine` tap fills a ring buffer with PCM.
2. **Whisper is not a streaming model** — it transcribes a window. That is why `listen()`
   defaults to **Nemotron-streaming** (2.5 GB, iOS), which is, and why the catalog carries it.
3. Partial results come from re-transcribing the growing tail of the buffer. That is why
   `isFinal == false` means *this text will be replaced*, not *this text is incomplete*.
4. Finalising a segment needs **voice-activity detection** — a pause is what ends an utterance.
5. Diarization runs over a longer window and **back-fills** speaker labels, which is why
   `speaker` is optional and arrives late. An utterance can be delivered final with
   `speaker == nil` and be updated afterwards.

Point 5 has a consequence the API must not hide: with `diarize: true`, an already-final
utterance can change its speaker. The stream therefore re-emits it. A view keyed by
`Utterance.id` updates in place; one that blindly appends will duplicate. Document this at the
call site, not in a footnote.

## What has to be built

Every model already exists. Only the plumbing is missing.

| Piece | State | Notes |
|---|---|---|
| Ring buffer + `AsyncStream` over the mic | new | extends `MicRecorder`, which today only does start/stop |
| **Voice-activity detection** | **new** | the only genuinely new component; small, no model, no dependency |
| Speaker back-fill | new | `KitDiarizer` exists; the re-emit logic does not |
| `say` | thin | `KitSpeaker` exists; add playback |
| `transcribeStream(file:)` | new | same pipeline as `listen()`, fed from a file instead of the mic |

## The dependency that decides whether any of this is adoptable

On iPhone today the speech stack is Nemotron-streaming 2.5 GB + VibeVoice 1.4 GB +
Sortformer 451 MB ≈ **4.3 GB**. The light models are macOS-only: Parakeet 1.3 GB and Kokoro
341 MB have no iOS variant published.

With those two ported, the same stack is ≈ **2.1 GB**. Both models are already converted; only
the iOS variant is missing. **Nothing else on the map moves adoption cost that far for that
little work**, and a 4.3 GB first-use download is not something an engineer adds on a Tuesday
afternoon without asking anyone — which is the adoption pattern this is aiming at.

Treat the iOS ports as a precondition for the speech product, not as a follow-up.

## Open questions

- **Does the streaming model hold quality at short windows?** Nemotron-streaming is built for it,
  but partial-then-final rewriting is visible to the user and jarring if the text jumps. Needs a
  measured look at how much the tail changes between partials.
- **What is the VAD threshold?** Too eager and utterances fragment; too slow and the app feels
  unresponsive. Should be measured, not guessed, and probably exposed.
- **Does `listen()` need a language argument?** `transcribe` has one. Auto-detection on a live
  stream is a different problem from auto-detection on a file.
- **Battery.** A live pipeline runs the GPU continuously. Nothing on this map has been measured
  for sustained power draw, and a voice feature that drains a phone in an hour will be removed by
  whoever ships it.
