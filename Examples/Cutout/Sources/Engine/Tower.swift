// Tower — the vision tower with the mask decoder's taps (`wevisone_vl_2b_vision_taps_g14`): one
// stateless function, patches fp16 [784, 1536] → image_embeds fp16 [196, 2048] (the decoder's image rows),
// deepstack_embeds fp16 [588, 2048] (the three deepstack layers' rows) and taps fp16 [4, 1024, 28, 28] (the ViT's
// layers 5 / 11 / 17 / 23 before the merger, row-major on the 28 × 28 patch grid: the mask decoder's visual input).
// The outputs stay NDArrays: the decoder and graph A read them as they come.

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public final class VisionTaps: @unchecked Sendable {
    public struct Output: @unchecked Sendable {
        public let imageEmbeds: NDArray
        public let deepstackEmbeds: NDArray
        public let taps: NDArray
    }

    public let url: URL
    public let loadSeconds: Double
    public let descriptor: [String: Any]
    private let function: InferenceFunction
    private let patchesDescriptor: NDArrayDescriptor

    public init(contentsOf url: URL, options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        guard let name = model.functionNames.first, let fd = model.functionDescriptor(for: name),
              let fn = try model.loadFunction(named: name)
        else { throw CutoutError.contract("tower \(url.lastPathComponent): no function") }
        loadSeconds = secondsSince(t0)
        let want: [String: TensorSpec] = [
            "image_embeds": TensorSpec(shape: [196, 2048], type: .float16),
            "deepstack_embeds": TensorSpec(shape: [588, 2048], type: .float16),
            "taps": TensorSpec(shape: [4, 1024, 28, 28], type: .float16),
        ]
        guard fd.stateNames.isEmpty, fd.inputNames == ["patches"],
              let pin = ND.descriptor(fd.inputDescriptor(of: "patches")), pin.shape == [784, ImageHost.patchVector],
              pin.scalarType == .float16, Set(fd.outputNames) == Set(want.keys),
              want.allSatisfy({ TensorSpec.of(fd.outputDescriptor(of: $0.key)) == $0.value })
        else {
            throw CutoutError.contract(
                "tower \(url.lastPathComponent) is not patches f16 [784, 1536] -> image_embeds, deepstack_embeds, taps: \(describe(fd))")
        }
        self.url = url
        self.function = fn
        self.patchesDescriptor = pin
        self.descriptor = describe(fd)
    }

    public func encode(patches: [Float16]) async throws -> Output {
        let input = ND.make(patches, patchesDescriptor)
        var outputs = try await function.run(inputs: ["patches": input])
        guard let e = outputs.remove("image_embeds")?.ndArray, let d = outputs.remove("deepstack_embeds")?.ndArray,
              let t = outputs.remove("taps")?.ndArray
        else { throw CutoutError.contract("tower: an output is missing") }
        return Output(imageEmbeds: e, deepstackEmbeds: d, taps: t)
    }
}
