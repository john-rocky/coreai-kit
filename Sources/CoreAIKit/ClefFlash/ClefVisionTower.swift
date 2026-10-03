// From the model zoo's apps/ClefFlash/Sources/ClefFlash/VisionTower.swift (5ef2247, sha256 4f1b3e8d1d97, the code its Swift gate ran), identifiers prefixed Clef for the kit.
// ClefVisionTower — one fixed-grid tower graph (`.aimodel` for the runtime's JIT, `.aimodelc` for an AOT asset):
// patches float32 [4 G^2, 1536] -> image_embeds float32 [G^2, 4096], G = 8 (g256) or 14 (g448). Stateless.

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
final class ClefVisionTower: @unchecked Sendable {
    static let hidden = 4096

    let grid: Int
    let url: URL
    /// AIModel(contentsOf:) + loadFunction, in seconds.
    let loadSeconds: Double
    let descriptor: ClefJSON
    private let function: InferenceFunction
    private let patchesDescriptor: NDArrayDescriptor
    private let outputName = "image_embeds"

    init(contentsOf url: URL, grid: Int, options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        guard let name = model.functionNames.first, let fd = model.functionDescriptor(for: name),
              let fn = try model.loadFunction(named: name)
        else { throw ClefFlashError.contract("tower \(url.lastPathComponent): no function") }
        loadSeconds = clefSecondsSince(t0)
        let n = 4 * grid * grid
        try clefCheckFunction(fd, what: "tower \(url.lastPathComponent)",
                          inputs: ["patches": ClefTensorSpec(shape: [n, ClefImagePreprocess.patchVector], type: .float32)],
                          outputs: [outputName: ClefTensorSpec(shape: [grid * grid, Self.hidden], type: .float32)],
                          states: [:])
        self.grid = grid
        self.url = url
        self.function = fn
        self.patchesDescriptor = ClefND.descriptor(fd.inputDescriptor(of: "patches"))!
        self.descriptor = clefDescribe(fd)
    }

    /// patches [4 G^2 * 1536] -> image_embeds [G^2 * 4096], row-major.
    func encode(patches: [Float]) async throws -> [Float] {
        let input = ClefND.make(patches, patchesDescriptor)
        var outputs = try await function.run(inputs: ["patches": input])
        guard let array = outputs.remove(outputName)?.ndArray else {
            throw ClefFlashError.contract("tower: no \(outputName) in the outputs")
        }
        let emb = ClefND.read(array, as: Float.self)
        guard emb.count == grid * grid * Self.hidden else {
            throw ClefFlashError.contract("tower: \(emb.count) values for \(grid * grid) x \(Self.hidden)")
        }
        return emb
    }
}
