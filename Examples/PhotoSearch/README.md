# PhotoSearch

Semantic photo search, fully on-device: the photo library is indexed with CLIP image
embeddings (Neural Engine, ~4 ms per photo), and free-text queries like
"red bike at the beach" rank photos by cosine similarity.

Built on `CoreAIKitVision` — the whole ML surface of this app is
`encoder.encode(image:)` / `encoder.encode(text:)`.

## Run

```bash
xcodegen generate
open PhotoSearch.xcodeproj
```

Run on an iPhone (photo library + Neural Engine). First launch downloads the CLIP bundle
(~290 MB) from the Hugging Face Hub, then indexes the library once; embeddings persist
under Application Support, so later launches only index new photos.

## Notes

- Brute-force search (one dot product per photo via vDSP) — instant up to tens of
  thousands of photos; no vector database needed.
- Index storage is a flat Float32 file + an id list, ~2 KB per photo.
