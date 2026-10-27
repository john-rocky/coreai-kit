# Cutout — cut out what you name in a photo, with its soft edge, on the device

Pick one photo and type what to cut out — "the dog", "the red truck", "people": on the device, WeVisOne-VL-2B finds each
thing that matches, draws its box and its alpha matte down to the half-transparent edge, and Save writes the cut-out as
a PNG with the background removed (to Photos on an iPhone, to a file on a Mac); Lined up sets the pieces side by side on
one sheet, and Save then writes that sheet. On an iPhone 18 Pro (iOS 27.2) one match took about 1 s (0.99–1.00 s) and
seven matches 4.1 s, measured 2026-10-25. The photo never leaves the device. One screen: a photo and a few words in, the
boxes, the cut-out and the pieces lined up out; the line under it says how long the boxes and the mattes took, and where
they ran.
A 14.6 s clip of the app on an iPhone 18 Pro — a kitten and a puppy, "the kitten", only the kitten cut out and saved
to Photos — is
[in coreai-assets](https://github.com/john-rocky/coreai-assets/blob/main/demos/wevisone-vl-2b-cutout-iphone-20261027.mp4).

The model is [WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B) by WeChat Vision, Tencent Inc. (revision
`3dad9ce8`): built upon Qwen3-VL, it answers detection, referring, counting, pointing, OCR, segmentation and alpha matting
as structured text, and a 9.5M-parameter pixel decoder turns each object the model names into a matte. Its Core AI port
is four graphs: the text decoder (int8, 2.46 GB), the vision tower with the pixel decoder's inputs (fp16, 0.81 GB) and
the pixel decoder in two parts (fp16, 20 MB). The take-home is [`Sources/QuickStart.swift`](Sources/QuickStart.swift):
one function, a photo and the words in, each match's box and matte and the PNG out.

This sample runs the model on the system CoreAI framework itself ([`Sources/Engine/`](Sources/Engine/)), not through
CoreAIKit: the pixel decoder needs the decoder's hidden state at each object the model writes, and the kit's pipelined
engine hands back token ids only. It needs no package but swift-transformers' tokenizer.

## Run it with your own photo

```sh
cd Examples/Cutout
tools/fetch_examples.sh              # the three example photos (two CC0, one NASA public domain)
export DEVELOPMENT_TEAM=XXXXXXXXXX   # your team id; or pick your team in Signing & Capabilities
xcodegen generate                    # brew install xcodegen
open Cutout.xcodeproj                # choose your iPhone (iOS 27) or My Mac and Run; the scheme runs Release
```

The app does not download the model. Take the folder for your device from the Hugging Face repository
[`mlboydaisuke/WeVisOne-VL-2B-CoreAI`](https://huggingface.co/mlboydaisuke/WeVisOne-VL-2B-CoreAI) and name it
`WeVisOne-VL-2B-CoreAI`:

```sh
hf download mlboydaisuke/WeVisOne-VL-2B-CoreAI --include "ios/*" --local-dir wevisone    # iPhone, 3.30 GB
mv wevisone/ios WeVisOne-VL-2B-CoreAI
hf download mlboydaisuke/WeVisOne-VL-2B-CoreAI --include "macos/*" --local-dir wevisone  # Mac, 10.19 GB
mkdir -p ~/Library/Application\ Support/Cutout
mv wevisone/macos ~/Library/Application\ Support/Cutout/WeVisOne-VL-2B-CoreAI
```

On an iPhone, put the folder in the app's Documents (Finder › your iPhone › Files › Cutout, or
`xcrun devicectl device copy to … --domain-type appDataContainer`); on a Mac it goes in
`~/Library/Application Support/Cutout/`; or pass `-models <folder>` at launch.

1. The app loads the model as it starts. READY means it can cut out.
2. Take a photo, Pick a photo, or tap one of the three examples (a tap also puts its words in the box); type what to cut
   out — "the dog", "the red truck", "people" — or tap a word under the box; Cut out.
3. The cut-out appears over a checkerboard; Photo | Cut out | Lined up switches between it, the photo with each match's
   box drawn on it, and the pieces apart: each cropped to its matte and set side by side on one sheet at the photo's
   scale, bottoms on one line (more than four pieces wrap into rows). When several things match, each gets a number;
   tap one to leave it out. Save to Photos (on a Mac, Save PNG…) writes the chosen pieces from the photo at its full
   size: the cut-out, or the lined-up sheet while Lined up is showing.

On an iPhone 18 Pro (iOS 27.2), cutting out the lighthouse took 0.99–1.00 s from Cut out to the finished cut-out (the
boxes in 0.47–0.48 s — the vision tower 0.09 s, the 237-token prompt 0.14 s, the 13-token answer 0.23 s — and the
matte in 0.49–0.51 s, 0.39–0.41 s of it the second mask graph); the seven light chess pieces took 4.1 s, 2.8 s of it the
second mask graph (0.4 s a piece). The first launch after the install reached READY 6.9 s after it started and took 8.1 s for
its first request (the phone specializes each graph once); later launches reached READY in 1.5–2.6 s. The app's memory
footprint peaked at 2.0 GB. Measured on 2026-10-25, warm, three requests each. On a Mac M4 Max (macOS 27.0, the
ahead-of-time graphs), one thing in a 3744 × 5616 NASA photo (decoded at 2731 × 4096) took 0.55 s (boxes 0.35 s, matte
0.19 s; the PNG another 0.29 s); "people" in another NASA photo, three matches, took 0.66 s.

The app needs the entitlement `com.apple.developer.kernel.increased-memory-limit` (`Cutout.entitlements`): the 2.46 GB
decoder, the 0.81 GB tower and the two mask graphs specialize in one process.

## The screen

Top to bottom: the state (LOADING, READY, WORKING, DONE, NONE, STOPPED, FAILED) and what is happening; the photo with
the boxes, the cut-out, or the pieces lined up; the three examples, Pick a photo and Take a photo; what to cut out, and
words to tap; Cut out | Boxes only; Cut out, Save and Stop; the pieces found (or the model's answer when it found
nothing); the numbers line; the model, the device, the OS and the app version.

Boxes only asks the model to find the things (the package's detection prompt) and draws their boxes, without mattes.
Pick a photo opens the system photo picker, which runs outside the app and needs no permission. Take a photo asks for
the camera the first time. Save to Photos asks for permission to add to the library (the app never reads it).

## The numbers line

```
boxes 0.45 s · matte 0.18 s · 1 found
GPU · on-device
```

- boxes: from Cut out to the end of the answer — the 448 × 448 resize, the vision tower, the prompt and the generated
  tokens. The model is loaded before READY and is not in it.
- matte: the 1024 × 1024 resize and the two mask graphs for every piece.
- found: the pieces with a box and a matte.

## What one request does

The steps of the package's `predict(task="segmentation")` (its `matting` task shares the prompt and the mattes), with
the photo stretched to a fixed 448 × 448 grid (the package keeps the aspect ratio inside 2025 × 32² pixels; a fixed grid
keeps the graphs static):

1. The photo, decoded with ImageIO (JPEG, HEIC, PNG …, the first image), turned upright by its EXIF orientation (a long
   side over 4096 px is subsampled while it decodes).
2. Pillow's bicubic resize to 448 × 448 and to 1024 × 1024, written out in integer arithmetic: the same bytes as Pillow
   on PNG input (JPEG decoders differ by a little, ImageIO's from Pillow's libjpeg).
3. The vision tower: the image rows, the deepstack rows and the pixel decoder's four feature maps.
4. The prompt, as the package builds it: "Locate the instances that match the following description and provide the
   alpha mattes: the dog." in the checkpoint's chat template, the system line "You are a helpful assistant.".
5. The decoder, greedy: the prompt in 64-token calls (the last one padded), then one token per call until the end of
   the answer (at most 1,024 positions in all, the iOS limit of a dynamic cache); the hidden state of each fed `<OBJ>`.
6. The boxes, by the package's `parse_seg_boxes`: the four coordinates (per mille) before each `<OBJ>`.
7. The pixel decoder: each box's Fourier encoding made on the app's CPU in fp32 (it has to equal PyTorch's bit for bit;
   see `Sources/Engine/BoxFourier.swift`), part A once per 16 pieces, part B once per piece.
8. Sigmoid, then a bilinear resize to the photo's size: the matte. The PNG's alpha is the chosen mattes added and
   clipped to 1.

## The model folder

```
WeVisOne-VL-2B-CoreAI/
├── wevisone_vl_2b_decode_int8hu_pf64/          metadata.json, tokenizer/, wevisone_vl_2b_decode_int8hu_pf64.aimodel
├── wevisone_vl_2b_vision_taps_g14.aimodel
├── wv_mask_a_g28_n16_fp16_hf.aimodel
└── wv_mask_b_fp16.aimodel
```

On an iPhone the app specializes the `.aimodel` files on the device the first time (3.30 GB). On a Mac it opens an
ahead-of-time copy beside each (`<name>.h16c.aimodelc`, 10.19 GB: the decoder's int8 Linears get an fp16 copy) when one
is there. A `cutout.json` in the folder can name other parts (`{"decoder": "…", "tower": "…", "graph_a": "…",
"graph_b": "…"}`). The Hugging Face repository `mlboydaisuke/WeVisOne-VL-2B-CoreAI` holds the folder twice: `ios/` (the
four `.aimodel` graphs) and `macos/` (the same four compiled ahead of time for the Mac). How each graph was made and
checked against the original model: the Core AI model zoo's `models/wevisone-vl-2b` card.

## Autoplay and the record

For recordings and checks from a script, the app taps its own controls with launch arguments:

```sh
xcrun devicectl device process launch --device <udid> com.coreaikit.cutout -- \
  -autoplay 1 -example lighthouse -save 1 -frames 1 -log 1
```

| argument | meaning |
|---|---|
| `-autoplay 1` | wait for READY, then tap the example (or open `-image <file>`), put `-words "…"` in the box, Cut out, wait for the answer |
| `-example lighthouse` | `lighthouse`, `chess` or `airplane` |
| `-task boxes` | Boxes only |
| `-view photo` | leave the boxes on the screen instead of the cut-out |
| `-lineup 2.5` | switch to Lined up 2.5 s after the answer (`-save 1` then saves the sheet) |
| `-save 1` | tap Save at the end (on a Mac, `-saveto <dir>` instead of the save panel) |
| `-frames 1` | write the screen as PNGs to `Documents/frames/<epoch>/`: READY, the pick, the words, the answer, Lined up, the save |
| `-log 1` | append the steps to `Documents/cutout-autoplay.log` and write `Documents/cutout-run-<epoch>-<ms>.json` per request: the step times, the answer, the boxes, the memory footprint, the thermal state |
| `-models <folder>` | the model folder (a path, or a name in Documents) |
| `-docs <dir>` | where the log, the records and the frames go instead of Documents |
| `-quit 1` | exit after the record |
| `-bench 3` | the example request once and 3 times again, through Cut out |
| `-fixture <rows.json>` | rows of photos and prompts through the request code, the outputs per row under `Documents/fixture/` (`Sources/Engine/Fixture.swift`) |
| `-tail 1` | the prompt's last tokens fed both ways, padded and one by one, timed (`WeVisOneDecoder.tailBench`) |
| `-photosauth denied` | Save behaves as if Photos access were off (the screen's sentence for that case; the record says it was set this way) |

`swift run cutout-cli run --models <folder> --image x.jpg --words "the dog" --out <dir>` does one request on a Mac
without Xcode (`CLI/main.swift`; `--lineup` also writes the lined-up sheet; also `fixture`, `tail`, `bench` and
`boxpe`). `UITests/CutoutUITests.swift` runs the
lighthouse example and answers the system's Photos request (allow, or decline); on a phone the first UI test session
needs the passcode for UI automation, and unattended it stops with "Timed out while enabling automation mode".

## When it does not cut out

| what happens | what the screen says |
|---|---|
| nothing in the photo matches the words (the model answers `<box>none</box>` or no box) | NONE · Nothing in this photo matched “the giraffe” (and the answer as written) |
| the model folder is not there | FAILED · The model files are not on this device: no folder WeVisOne-VL-2B-CoreAI. Put the model folder there (README: Run it). |
| a file of the model is missing | FAILED · The model files are not complete: … is missing … |
| the first Save | the system asks "Allow “Cutout” to add photos and videos to your photo library?"; the screen says Saving… until it is answered |
| Save without Photos access | Photos access is off: Settings › Apps › Cutout › Photos |
| Cut out without a photo | Choose a photo first: tap an example, pick or take one |
| Cut out with no words | Type what to cut out, or tap a word under the box |
| a picked photo the app cannot decode | Could not read that photo |
| every piece left out, then Save | Choose at least one piece to save |
| Stop during a request | STOPPED |
| the system ends the app for its memory (a phone without the entitlement, or other apps holding memory) | the app closes; the record of the request is not written |

These are the app's own sentences, from `CutoutModel.swift` and `Sources/Engine/`.

Saving to Photos: on the iPhone 18 Pro, with access allowed, Save added the lighthouse cut-out to Photos (a PNG of
1,187,126 bytes) and the screen said "Saved to Photos" (2026-10-27). The sentence for a refusal was checked with
`-photosauth denied`. The system's request stays on the screen until someone answers it, over other apps too. On a
Mac, Save wrote the file (with `-save 1 -saveto <dir>`: the lighthouse, a
1024 × 683 PNG with its alpha).

## Example photos

Three photos, none showing a person, a brand or a logo. They are not in the repository: `tools/examples.json` names each
one's URL, size and sha256, and `tools/fetch_examples.sh` downloads them into `Examples/` (the folder the app bundles) and
checks both numbers; `--check` checks without the network.

| id | shows | file | licence | source page |
|---|---|---|---|---|
| `lighthouse` | a white lighthouse with a red base on rocks by the sea (Examples/Look's `scene`) | `lighthouse.jpg`, 1024 × 683 | CC0 1.0 | [Flickr](https://www.flickr.com/photos/66741228@N05/33074457825) |
| `chess` | light wooden chess pieces on a worn chessboard, close up (Examples/Look's `object`) | `chess.jpg`, 960 × 640 | CC0 1.0 | [StockSnap](https://stocksnap.io/photo/chess-board-NUAITIXILP) |
| `airplane` | NASA's Super Guppy approaching a runway at Kennedy Space Center, 2017 | `airplane.jpg`, 853 × 1280 | public domain (NASA) | [NASA](https://images.nasa.gov/details/KSC-20170424-PH_SWW01_0007) |

Measurement hooks, off unless set: `-tbench`, `-bbench` and `-abench` time another tower, graph B or graph A from the
model folder against the loaded one (alternating in one process), and the environment variables `CUTOUT_TOWER`,
`CUTOUT_GRAPH_A`, `CUTOUT_GRAPH_B` and `CUTOUT_GRAPH_A_UNIT=ane` load another part for one launch; without them the app
does what this page says.
