// Screenshot.swift — a headless screenshot hook for the Mac, in the same spirit as the
// SPOTLIGHT_SELFTEST gate: launch with SPOTLIGHT_SHOT=/path/out.png (optionally SPOTLIGHT_AUTOASK)
// and the app loads, answers the first sample question, renders its own window to a PNG, and exits.
// It renders the view hierarchy in-process (`cacheDisplay`), so it needs no Screen Recording
// permission and no UI automation — useful for verifying the layout without a hands-on session.

#if os(macOS)
import AppKit

enum Screenshotter {
    /// Render the frontmost window's content view to a PNG on disk. In-process view caching, so no
    /// system capture permission is involved.
    @MainActor
    static func capture(to path: String) {
        guard
            let window = NSApp.windows.first(where: { $0.isVisible }) ?? NSApp.windows.first,
            let view = window.contentView,
            let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds)
        else { return }
        view.cacheDisplay(in: view.bounds, to: rep)

        // `cacheDisplay` renders the view layer only — the window's background material is not
        // included, leaving a translucent backdrop. Composite onto an opaque window-colored fill
        // so the saved PNG matches what's on screen.
        let bounds = view.bounds
        let opaque = NSImage(size: bounds.size)
        opaque.lockFocus()
        NSColor.windowBackgroundColor.setFill()
        NSRect(origin: .zero, size: bounds.size).fill()
        rep.draw(in: NSRect(origin: .zero, size: bounds.size))
        opaque.unlockFocus()

        guard
            let tiff = opaque.tiffRepresentation,
            let flattened = NSBitmapImageRep(data: tiff),
            let data = flattened.representation(using: .png, properties: [:])
        else { return }
        try? data.write(to: URL(fileURLWithPath: path))
    }
}
#endif
