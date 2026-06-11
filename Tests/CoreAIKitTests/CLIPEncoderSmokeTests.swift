import CoreGraphics
import XCTest

@testable import CoreAIKitVision

/// End-to-end smoke over a real CLIP bundle. Opt-in (loads a model, runs the GPU):
///
///     KIT_CLIP_BUNDLE=/path/to/dir swift test --filter CLIPEncoderSmoke
///
/// The bundle directory holds one *.aimodel plus an HF tokenizer.json.
final class CLIPEncoderSmokeTests: XCTestCase {
    private func solidImage(red: CGFloat, green: CGFloat, blue: CGFloat) throws -> CGImage {
        let size = 256
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: nil, width: size, height: size, bitsPerComponent: 8,
                bytesPerRow: size * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw VisionError.imageRenderFailed
        }
        ctx.setFillColor(CGColor(srgbRed: red, green: green, blue: blue, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        guard let image = ctx.makeImage() else { throw VisionError.imageRenderFailed }
        return image
    }

    private func norm(_ v: [Float]) -> Float {
        var sum: Float = 0
        for x in v { sum += x * x }
        return sum.squareRoot()
    }

    func testProbeTokenizerAndZeroShot() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_CLIP_BUNDLE"] else {
            throw XCTSkip("Set KIT_CLIP_BUNDLE to a local CLIP bundle directory to run.")
        }
        let bundle = URL(fileURLWithPath: path)

        // Probe: surface the graph contract for the record (verification item 5).
        let aimodel = try FileManager.default.contentsOfDirectory(
            at: bundle, includingPropertiesForKeys: nil
        ).first { $0.pathExtension == "aimodel" }!
        let graph = try await GraphModel(contentsOf: aimodel, computeUnits: .gpu)
        for name in graph.inputNames {
            print("PROBE input \(name): \(graph.shape(ofInput: name) ?? [])")
        }
        for name in graph.outputNames {
            print("PROBE output \(name): \(graph.shape(ofOutput: name) ?? [])")
        }

        // Tokenizer parity against the reference CLIP encoding.
        let tokenizer = try CLIPTokenizer(bundleAt: bundle)
        XCTAssertEqual(
            Array(tokenizer.encode("a photo of a cat").prefix(7)),
            [49406, 320, 1125, 539, 320, 2368, 49407])

        // Zero-shot sanity: in-graph L2 norm + color discrimination.
        let encoder = try await ImageTextEncoder(bundleAt: bundle, computeUnits: .gpu)
        XCTAssertEqual(encoder.embeddingDimension, 512)

        let redImage = try await encoder.encode(image: solidImage(red: 1, green: 0, blue: 0))
        let blueImage = try await encoder.encode(image: solidImage(red: 0, green: 0, blue: 1))
        let redText = try await encoder.encode(text: "a red square")
        let blueText = try await encoder.encode(text: "a blue square")

        XCTAssertEqual(norm(redImage), 1.0, accuracy: 0.05)
        XCTAssertEqual(norm(redText), 1.0, accuracy: 0.05)

        let rr = ImageTextEncoder.cosineSimilarity(redImage, redText)
        let rb = ImageTextEncoder.cosineSimilarity(redImage, blueText)
        let bb = ImageTextEncoder.cosineSimilarity(blueImage, blueText)
        let br = ImageTextEncoder.cosineSimilarity(blueImage, redText)
        print("PROBE cos red·'red square'=\(rr) red·'blue square'=\(rb)")
        print("PROBE cos blue·'blue square'=\(bb) blue·'red square'=\(br)")
        XCTAssertGreaterThan(rr, rb, "red image should match 'a red square' better")
        XCTAssertGreaterThan(bb, br, "blue image should match 'a blue square' better")
    }
}
