# Meeting — who said what, fully on-device

Speaker-attributed transcription: [Streaming Sortformer v2](https://huggingface.co/mlboydaisuke/Streaming-Sortformer-Diar-CoreAI)
(up to 4 speakers) or [Nemotron-3-Diarization](https://huggingface.co/mlboydaisuke/Nemotron-3-Diarization-CoreAI)
(up to 8) segments the clip into speaker turns ("who spoke when"), then the on-device ASR you
pick (Whisper / Qwen3-ASR / Parakeet, or Apple's) transcribes each turn. Two models behind one call:

```swift
import CoreAIKit

let meeting = try await MeetingTranscriber(asr: "whisper-large-v3-turbo")
let transcript = try await meeting.transcribe(samples: try AudioFile.pcm16kMono(url))
print(transcript.text)
// Speaker 1 [0.3–4.1s]: With her white paint and her scarlet smokestack…
// Speaker 2 [4.6–6.3s]: …
```

## Run it

```bash
cd Examples/Meeting
swift run meeting-cli --audio meeting.wav                          # Whisper turns
swift run meeting-cli --audio meeting.wav --asr parakeet-tdt-0.6b-v3
swift run meeting-cli --audio meeting.wav --diarizer nemotron-3-diarization --asr system --locale en-US
swift run meeting-cli --list-models                                # what's in the catalog
```

First run downloads the diarizer (226 MB Sortformer, 200 MB Nemotron-3-Diarization) and the ASR;
`--asr system` is Apple's transcriber, which the app does not download. Afterwards everything is
local — airplane mode works. Audio is anything AVFoundation reads (wav/m4a/mp3/…), resampled to
16 kHz mono.

`diarize-gate` checks the diarizer alone against a reference's per-frame probabilities: agreement
at 0.5, the turns, and the wall time. With the model zoo's golden files for Nemotron-3-Diarization
it reproduces the model card's agreement:

```bash
swift run -c release diarize-gate --audio clip.wav --golden clip_ll_probs.f32le
```

The pieces are separately usable: `KitDiarizer` alone answers "who spoke when"
(`diarize(samples:)` → `[SpeakerSegment]`), and `MeetingTranscriber(diarizer:transcriber:)`
composes any diarizer with any ASR you already loaded.

For a GUI take on the same pipeline, see the coreai-audio app's Transcribe tab
("Diarize — who said what") in the zoo repo.
