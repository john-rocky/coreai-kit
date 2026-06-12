import CoreGraphics
import XCTest

@testable import CoreAIKitVision

/// End-to-end smoke over a real Depth Anything bundle. Opt-in:
///
///     KIT_DEPTH_BUNDLE=/path/to/dir swift test --filter DepthEstimatorSmoke
final class DepthEstimatorSmokeTests: XCTestCase {
    /// A scene-ish synthetic image: horizon gradient + a dark block "object".
    private func syntheticScene() throws -> CGImage {
        let size = 448
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: nil, width: size, height: size, bitsPerComponent: 8,
                bytesPerRow: size * 4, space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw VisionError.imageRenderFailed
        }
        for y in 0..<size {
            let shade = CGFloat(y) / CGFloat(size)
            ctx.setFillColor(CGColor(srgbRed: 0.4 + shade * 0.5, green: 0.6, blue: 1.0 - shade * 0.6, alpha: 1))
            ctx.fill(CGRect(x: 0, y: y, width: size, height: 1))
        }
        ctx.setFillColor(CGColor(srgbRed: 0.1, green: 0.08, blue: 0.06, alpha: 1))
        ctx.fill(CGRect(x: size / 3, y: size / 8, width: size / 3, height: size / 2))
        guard let image = ctx.makeImage() else { throw VisionError.imageRenderFailed }
        return image
    }

    func testProbeAndDepthSanity() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_DEPTH_BUNDLE"] else {
            throw XCTSkip("Set KIT_DEPTH_BUNDLE to a local depth bundle directory to run.")
        }
        let bundle = URL(fileURLWithPath: path)

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

        let estimator = try await DepthEstimator(bundleAt: bundle, computeUnits: .gpu)
        let map = try await estimator.estimateDepth(for: syntheticScene())
        print("PROBE depth map \(map.width)x\(map.height)")

        XCTAssertGreaterThan(map.width, 0)
        XCTAssertGreaterThan(map.height, 0)
        XCTAssertEqual(map.values.count, map.width * map.height)
        XCTAssertTrue(map.values.allSatisfy(\.isFinite))

        // Non-degenerate output: a structured scene must not produce constant depth.
        let mean = map.values.reduce(0, +) / Float(map.values.count)
        let variance = map.values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
            / Float(map.values.count)
        print("PROBE depth mean=\(mean) variance=\(variance)")
        XCTAssertGreaterThan(variance, 0)

        XCTAssertNotNil(map.cgImage())
    }
}
