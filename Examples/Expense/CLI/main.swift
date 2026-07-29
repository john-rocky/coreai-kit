// expense-cli — the receipt flow with no Xcode and no device.
//
//   swift run expense-cli --image receipt.jpg
//   swift run expense-cli --image receipt.jpg --json
//
// Progress goes to stderr so stdout stays machine-checkable.

import CoreAIOps
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func fail(_ s: String) -> Never { err(s); exit(1) }

let usage = "usage: expense-cli --image <path> [--json]"

var imagePath: String?
var asJSON = false
var args = Array(CommandLine.arguments.dropFirst())
while let a = args.first {
    args.removeFirst()
    switch a {
    case "--image": imagePath = args.isEmpty ? nil : args.removeFirst()
    case "--json": asJSON = true
    case "-h", "--help": print(usage); exit(0)
    default: fail("unknown argument \(a)\n\(usage)")
    }
}
guard let imagePath else { fail(usage) }

guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: imagePath) as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(src, 0, nil)
else { fail("could not read an image from \(imagePath)") }

CoreAI.onDownload { p in err("  downloading \(p.currentFile): \(Int(p.fraction * 100))%") }

let receipt = try await scanReceipt(image)

if asJSON {
    let obj: [String: Any] = [
        "merchant": receipt.merchant, "total": receipt.total,
        "date": receipt.date, "category": receipt.category,
    ]
    let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: data, as: UTF8.self))
} else {
    print("\(receipt.merchant)  \(receipt.total)  \(receipt.date)  [\(receipt.category)]")
}
