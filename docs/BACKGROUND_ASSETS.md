# Downloading the models before the user asks

The first-run download is the largest job this package still hands back. `CoreAI.transcribe`
now costs nothing, but `caption` is 3.3 GB and `summarize` is 2.3 GB, and a user who taps a
button and waits four minutes has met the feature at its worst.

Apple's answer is [Background Assets][ba]: the OS downloads on your behalf while the app is
not running — during install, on Wi-Fi, on charge — and hands you the files. **A Swift package
cannot ship that for you**, because it requires an app extension (`BADownloaderExtension`)
with its own bundle identifier and entitlement, and extensions live in the app target.

What the package can do is tell you exactly what to enqueue, which is the part that otherwise
means reading `catalog.json` by hand.

[ba]: https://developer.apple.com/documentation/backgroundassets

## The plan for one op

```swift
import CoreAIOps

let entry = ModelCatalog.builtin.entry(id: CoreAI.defaultVisionModel)!
let plan = try await ModelStore.default.downloadPlan(for: entry.modelID!)

for item in plan {
    // item.url          — public, unauthenticated
    // item.sizeBytes    — for a progress total that is right before it starts
    // item.destination  — where ModelStore will look for it afterwards
}
```

`downloadPlan` reads file metadata from the Hub; it downloads nothing.

## Wiring it to Background Assets

In your app's downloader extension:

```swift
struct ModelDownloader: BADownloaderExtension {
    func backgroundDownload(_ manifest: BAAppExtensionInfo) async -> Set<BADownload> {
        guard let entry = ModelCatalog.builtin.entry(id: CoreAI.defaultVisionModel),
              let model = entry.modelID,
              let plan = try? await ModelStore.default.downloadPlan(for: model)
        else { return [] }

        return Set(plan.map { item in
            BAURLDownload(
                identifier: item.destination.lastPathComponent,
                request: URLRequest(url: item.url),
                essential: false,
                fileSize: Int(item.sizeBytes),
                applicationGroupIdentifier: "group.your.app",
                priority: .default)
        })
    }
}
```

**Do not write the files into `item.destination` yourself.** `ModelStore`'s contract is that a
bundle appears at its final path only when every file of it is present: a half-present bundle
poisons the content-keyed on-device compilation cache, and later loads fail until the cache is
wiped. Stage the files in your app group container, and on first launch move the complete set
into place — or simply call `CoreAI.prepare(_:)` and let the store do it, using the extension
only to warm the URL cache.

## Before any of that, ask whether you need to

```swift
switch await CoreAI.capability(.caption) {
case .ready:                     break            // already on the device
case .needsDownload(let bytes):  enqueue(bytes)   // 3.3 GB — worth doing early
case .needsSystemAssets:         break            // the OS's, not yours
case .insufficientStorage, .unsupportedDevice: hideFeature()
}
```

And to find out what a whole app will pull, before writing any of this:

```
swift run coreai-doctor path/to/YourApp
```

which scans the sources for op calls and totals the models behind them, counting a model
shared by three ops once.
