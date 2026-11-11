# Model downloads and storage

ModelStore uses [swift-huggingface](https://github.com/huggingface/swift-huggingface)
for Hub requests and file downloads.
Files download over HTTP by default.
The `Xet` package trait is off by default;
enable it to download large files over Xet, with HTTP fallback:

```swift
.package(url: "https://github.com/john-rocky/coreai-kit", exact: "<version>", traits: ["Xet"])
```

The trait forwards to swift-huggingface, so swift-xet is compiled only when an app
opts in.
The package manifest requires Swift 6.1 or later to declare the trait.
For private or gated repositories on `https://huggingface.co`, the client reads
credentials from the standard Hugging Face environment variables and token files,
including `HF_TOKEN` and the token saved by `hf auth login`.
The account must have access to the repository.
`ModelStore(hubBaseURL:)` selects a mirror; it does not receive those credentials.

Complete bundles stay in
`Application Support/CoreAIKit/Models/<org>/<name>/<revision>/<variant>/`,
independent of the endpoint.
ModelStore does not read or write the shared Hugging Face file cache,
so each file is stored once.
Each download stages all files before it installs the bundle with one rename.
Installed bundles are excluded from iCloud backup.
A transport failure can use a complete cached copy of the same variant under
another revision.
Concurrent callers share one download; only the first caller receives progress.
HTTP downloads, and Xet downloads when the trait is on, report byte progress during
the transfer.

A model with no path takes its platform's folder: `macos/` on a Mac, `ios/` on an iPhone.
`ios/` holds the JIT graphs, which every iPhone generation specializes on its first load.
A repository can also carry `ios-<arch>/`: the same bundle with the graphs compiled for one
generation (`ios-h18p/` for the iPhone 17 Pro, `ios-h19p/` for the 18 Pro).
On an iPhone, a model whose path is `ios`, left nil or named, downloads the folder for its own
architecture (`AIModel.deviceArchitectureName`) when the repository has one, and `ios/` when it
does not, at the cost of one more listing.
A copy already on disk is used, whichever folder it is.
Any other path downloads as named, and a Mac never looks for an architecture folder.
A repo with no `ios/` lists the architectures it has folders for in the variant's `architectures`,
so `CatalogEntry.unavailableReason` can tell an iPhone it has no bundle before anything downloads,
and `catalog:` initializers throw `unsupportedDevice` instead of starting the download.
The catalog's `sizeMB` measures `ios/`, or the largest `ios-<arch>/` when a repo has no `ios/`;
an `ios-<arch>/` beside an `ios/` can be larger, and
`ModelStore.remoteSize(of:)` gives the figure for the device it runs on.
