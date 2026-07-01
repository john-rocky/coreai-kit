# DocSearch — on-device visual document retrieval (ColModernVBERT)

A self-contained iPhone demo of `VisualDocumentRetriever`: a bundled corpus of page **images**
is encoded once with ColModernVBERT's document encoder, your typed query is encoded with the
query encoder, and pages are ranked by **MaxSim** (late interaction). No OCR — pages are matched
as pictures.

Six sample pages ship in the app (`Sources/Pages/doc_*.png`: revenue, headcount, invoice, menu,
safety, schedule). Type a question and the matching page should rank first.

## Build

```bash
cd Examples/DocSearch
xcodegen generate
open DocSearch.xcodeproj   # select your iPhone, Run
```

(Bundle id `com.coreaikit.docsearch`, team `MFN25KNUGJ` — adjust the team to yours.)

## Models: sideload (skip the download)

The retriever needs two fp16 bundles. Launch the app once (so its data container exists, device
unlocked), then sideload — `ModelStore` finds them locally and skips the network entirely:

```bash
SRC=/Users/majimadaisuke/code/ColModernVBERT-CoreAI/kit
DST="Library/Application Support/CoreAIKit/Models/mlboydaisuke/ColModernVBERT-CoreAI/main"
UDID=$(xcrun devicectl list devices | awk '/iPhone/{print $NF; exit}')   # or paste your UDID

xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier com.coreaikit.docsearch --source "$SRC/query" --destination "$DST/query"
xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier com.coreaikit.docsearch --source "$SRC/doc"   --destination "$DST/doc"
```

After copying, confirm on-device the layout is `…/main/query/<…>.aimodel` + `…/main/query/tokenizer`
and `…/main/doc/<…>.aimodel` (if `devicectl` nested it as `query/query`, re-copy with
`--destination "$DST"`). Relaunch the app — it indexes the 6 pages, then search.

Alternatively, if the Hub repo is public and laid out as `query/` + `doc/`, omit the sideload and
the app downloads on first launch.
