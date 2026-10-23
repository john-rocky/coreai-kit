// KitTimestralForecaster.swift — Timestral-209M probabilistic forecasting: a univariate series of any length →
// 21 quantiles (0.01 … 0.99) for every step of a horizon of up to one decode block (1,024 steps).
//
// One stateless Core AI graph per bucket (L context patches × h horizon patches, R rows per call) plus the host
// arithmetic in `TimestralHost` (scaling, the refine frame, the cap, symmetric averaging, the non-negative clip).
// The bucket, the rows and the input type come from each bundle's own function descriptor, so any exported form
// (fp32, fp16 compute, the Neural Engine form; R = 1 or 2) runs through the same code. With several buckets the
// smallest one that holds the series and the horizon runs; a series longer than every bucket keeps its most recent
// points.
//
// Where each graph runs comes from its form, read from the graph's name (the zoo names a bundle
// `timestral_209m_L<L>_h<h>_r<R>_<form>`): the Neural Engine form (`fp16p7-qkv4-tanh-conv`) computes in fp16 throughout
// and is right on the Neural Engine only — on the GPU its attention scores saturate at fp16's ±65,504 and the forecast
// comes out finite and wrong — so it runs on the Neural Engine; the other forms (fp32, fp16-attn32, fp16w32) run on the
// GPU. An explicit unit the Neural Engine form cannot compute right on is refused at load, and a graph whose output
// reaches fp16's limit (a form placed where it overflows) stops the forecast with an error.
//
//   let forecaster = try await KitTimestralForecaster(bundleAt: url)   // a graph, or a folder of graphs
//   let f = try await forecaster.forecast(series, horizon: 96)
//   f.median, f.quantile(0.1), f.quantile(0.9)                         // 96 steps each
//
// The weights' licence restricts who may use them (Modified MIT with a revenue condition; see the model's card).

import CoreAI
import CoreAIKitVision
import Foundation

/// A probabilistic forecast: one row of quantiles per horizon step.
@available(macOS 27, iOS 27, *)
public struct QuantileForecast: Sendable {
    /// The probability levels of the columns, low to high.
    public let levels: [Float]
    /// `horizon` rows × `levels.count` columns, each row non-decreasing.
    public let quantiles: [[Float]]
    /// What ran and how long it took.
    public let details: Details

    public struct Details: Sendable {
        /// The bucket that ran (context patches, horizon patches, rows per graph call).
        public let bucket: KitTimestralForecaster.Bucket
        /// Graph calls for this forecast: 1 for an R = 2 graph, 2 for an R = 1 graph.
        public let graphCalls: Int
        /// Points of the series the model read: the most recent ones, at most the bucket's capacity.
        public let contextUsed: Int
        /// Points left out from the start because the series is longer than the bucket holds.
        public let contextDropped: Int
        /// The compute unit the bucket's graph was loaded with: its form's own unless the caller gave one.
        public let computeUnits: GraphModel.ComputeUnits
        /// Wall time of the graph calls (inputs in, outputs copied out), milliseconds.
        public let graphMilliseconds: Double
        /// Wall time of the whole forecast, host arithmetic included, milliseconds.
        public let totalMilliseconds: Double
    }

    public var horizon: Int { quantiles.count }

    /// The median (level 0.5) of every step.
    public var median: [Float] { quantiles.map { $0[TimestralHost.medianIndex] } }

    /// The forecast at one of `levels` for every step; nil for a level the model does not output.
    public func quantile(_ level: Float) -> [Float]? {
        guard let k = levels.firstIndex(where: { abs($0 - level) < 1e-6 }) else { return nil }
        return quantiles.map { $0[k] }
    }
}

public enum KitTimestralForecasterError: Error, LocalizedError, Sendable, Equatable {
    /// The series has no points.
    case emptySeries
    /// None of the points the model reads is a number (every one is NaN or infinite): there is nothing to scale by.
    case noObservedValues(pointsRead: Int)
    /// The horizon is outside 1 … `maximum` steps (one decode block of the loaded buckets).
    case horizonOutOfRange(requested: Int, maximum: Int)
    /// The released forecaster's ValueError: the forecast came out with NaN values.
    case degenerateInput(nanCount: Int)
    /// No graph in the bundle folder, or one that does not take Timestral's inputs.
    case bundle(String)
    /// An explicit compute unit for a Neural Engine form graph other than the Neural Engine: there its fp16 arithmetic
    /// overflows and the forecast would be wrong without an error.
    case wrongComputeUnits(graph: String, requested: String)
    /// The graph's output reached fp16's limit (±65,504) or infinity: it did not compute right where it ran (the
    /// Neural Engine form gives this on the GPU). No forecast is returned.
    case saturatedOutput(graph: String, values: Int, ranOn: String)

    public var errorDescription: String? {
        switch self {
        case .emptySeries:
            return "The series is empty: give at least one number."
        case .noObservedValues(let n):
            return n == 1
                ? "The value the model reads is missing (NaN or infinite): there is nothing to forecast from."
                : "All \(n) values the model reads are missing (NaN or infinite): there is nothing to forecast from."
        case .horizonOutOfRange(let requested, let maximum):
            return "Horizon \(requested) is outside 1…\(maximum) steps."
        case .degenerateInput(let n):
            return "forecast produced \(n) NaN values; the input is degenerate"
        case .bundle(let s):
            return s
        case .wrongComputeUnits(let graph, let requested):
            return "\(graph) is the Neural Engine form and computes right only on the Neural Engine; on \(requested) "
                + "its fp16 arithmetic overflows and the forecast would be wrong. Load it without a compute unit "
                + "(each graph then runs on its own) or on the Neural Engine."
        case .saturatedOutput(let graph, let n, let ranOn):
            return "\(graph) on \(ranOn) returned \(n) values at fp16's limit (±65,504): the graph did not compute "
                + "right there (the Neural Engine form does this on the GPU), so no forecast is returned."
        }
    }
}

@available(macOS 27, iOS 27, *)
public final class KitTimestralForecaster: @unchecked Sendable {  // immutable after init
    /// The model's catalog id.
    public static let catalogID = "timestral-209m"

    /// One graph's fixed shape, read from its function descriptor.
    public struct Bucket: Sendable, Hashable, CustomStringConvertible {
        /// Context patches (32 steps each): the graph reads at most `contextPatches × 32` points.
        public let contextPatches: Int
        /// Horizon patches: the graph forecasts at most `horizonPatches × 32` steps.
        public let horizonPatches: Int
        /// Rows per call: 2 = the series and its mirror in one call, 1 = one call each.
        public let rows: Int
        /// The graph's input scalar type ("float32" or "float16").
        public let inputType: String

        public var contextCapacity: Int { contextPatches * TimestralHost.patch }
        public var maximumHorizon: Int { horizonPatches * TimestralHost.patch }
        public var description: String { "L\(contextPatches) h\(horizonPatches) R\(rows) \(inputType)" }
    }

    /// The loaded buckets, smallest first.
    public var buckets: [Bucket] { graphs.map(\.bucket) }
    /// The longest horizon any loaded bucket forecasts (≤ 1,024 steps).
    public var maximumHorizon: Int { graphs.map(\.bucket.maximumHorizon).max() ?? 0 }
    /// The most points any loaded bucket reads; a longer series keeps its most recent points.
    public var contextCapacity: Int { graphs.map(\.bucket.contextCapacity).max() ?? 0 }
    /// The quantile levels of every forecast.
    public var levels: [Float] { TimestralHost.levels }

    private let graphs: [TimestralGraph]

    // MARK: - Loading

    /// Loads Timestral by its catalog id; downloads on first use. `computeUnits` nil (the default) runs each graph on
    /// its form's unit (`computeUnits(forGraph:)`); a unit given here is used for every graph, and one the Neural
    /// Engine form cannot compute right on is refused.
    public convenience init(
        catalog id: String = KitTimestralForecaster.catalogID,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits? = nil,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .quantileForecasting)
        guard let model = entry.modelID else { throw CoreAIKitError.modelNotAvailableOnPlatform(id: id) }
        let root = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: root, computeUnits: computeUnits)
    }

    /// Loads a local bundle: one graph (`.aimodel` / `.aimodelc`), or a folder holding one graph per bucket. Of the
    /// forms of one graph the device loads the compiled one when it was compiled for this device, else the JIT one.
    /// `computeUnits` as in `init(catalog:)`: nil runs each graph on its form's unit.
    public convenience init(bundleAt url: URL, computeUnits: GraphModel.ComputeUnits? = nil) async throws {
        try await self.init(bundles: KitTimestralForecaster.graphURLs(in: url).map {
            (url: $0, computeUnits: computeUnits ?? KitTimestralForecaster.computeUnits(forGraph: $0))
        })
    }

    /// Loads one graph per bucket, each with its own compute preference. A Neural Engine form graph with any unit
    /// but the Neural Engine is refused before anything loads.
    public init(bundles: [(url: URL, computeUnits: GraphModel.ComputeUnits)]) async throws {
        guard !bundles.isEmpty else { throw KitTimestralForecasterError.bundle("No graph to load.") }
        for (url, units) in bundles { try KitTimestralForecaster.check(units, for: url) }
        var loaded: [TimestralGraph] = []
        for (url, units) in bundles {
            loaded.append(try await TimestralGraph(url: url, computeUnits: units))
        }
        graphs = loaded.sorted {
            ($0.bucket.contextPatches, $0.bucket.horizonPatches) < ($1.bucket.contextPatches, $1.bucket.horizonPatches)
        }
    }

    // MARK: - Where a graph runs

    /// The compute unit a graph's form computes right on, read from its name: the Neural Engine for the Neural Engine
    /// form (a form ending in `-conv`: `fp16p7-qkv4-tanh-conv`, fp16 throughout with 1×1-convolution projections), the
    /// GPU for any other form (fp32, fp16-attn32, fp16w32) and for a name outside the zoo's
    /// `timestral_209m_L<L>_h<h>_r<R>_<form>` (with or without an architecture: `….h19p.aimodelc`).
    public static func computeUnits(forGraph url: URL) -> GraphModel.ComputeUnits {
        isNeuralEngineForm(url) ? .neuralEngine : .gpu
    }

    /// The form in a graph's name — `fp16p7-qkv4-tanh-conv` for `timestral_209m_L64_h32_r1_fp16p7-qkv4-tanh-conv.aimodel`
    /// — or nil for a name outside the zoo's pattern.
    static func form(of graph: URL) -> String? {
        let stem = GraphBundle.stem(of: graph)
        guard let r = stem.range(of: #"_L[0-9]+_h[0-9]+_r[0-9]+_"#, options: .regularExpression),
            r.upperBound < stem.endIndex
        else { return nil }
        return String(stem[r.upperBound...])
    }

    static func isNeuralEngineForm(_ graph: URL) -> Bool { form(of: graph)?.hasSuffix("-conv") ?? false }

    /// Refuses a unit the graph's form does not compute right on: the Neural Engine form anywhere but the Neural
    /// Engine. A GPU form takes any unit (it does not fit the Neural Engine, which leaves it to the GPU).
    static func check(_ units: GraphModel.ComputeUnits, for graph: URL) throws {
        guard isNeuralEngineForm(graph), units != .neuralEngine else { return }
        throw KitTimestralForecasterError.wrongComputeUnits(graph: graph.lastPathComponent, requested: name(of: units))
    }

    /// "the Neural Engine", "the GPU", "the CPU": for messages.
    static func name(of units: GraphModel.ComputeUnits) -> String {
        switch units {
        case .neuralEngine: return "the Neural Engine"
        case .gpu: return "the GPU"
        case .cpu, .cpuOnly: return "the CPU"
        }
    }

    /// The graphs in `url`, one per graph name across its subfolders (`ios/x.aimodel` and `ios-h19p/x.h19p.aimodelc`
    /// are two forms of `x`), each picked by `GraphBundle`'s device rule among the forms of its name.
    static func graphURLs(in url: URL) throws -> [URL] {
        if GraphBundle.isGraph(url) { return [url] }
        let found = GraphBundle.graphs(in: url, searchSubdirectories: true)
        guard !found.isEmpty else {
            throw KitTimestralForecasterError.bundle("\(url.path) holds no .aimodel or .aimodelc graph.")
        }
        let byName = Dictionary(grouping: found) { GraphBundle.stem(of: $0) }
        var urls: [URL] = []
        for key in byName.keys.sorted() {
            let forms = byName[key]!
            guard let pick = GraphBundle.pick(forms) else { throw GraphBundle.unloadable(forms, in: url) }
            urls.append(pick)
        }
        return urls
    }

    // MARK: - Forecast

    /// Forecasts `horizon` steps after `series` (NaN or ±inf = a missing value). The forecast's details say which
    /// bucket ran and how much of a long series it read.
    public func forecast(_ series: [Float], horizon: Int) async throws -> QuantileForecast {
        try await run(series, horizon: horizon, trace: false).forecast
    }

    /// The forecast with the intermediates the parity gate compares with the Python host: the graph's inputs
    /// (patches of both rows, cos, sin, attn_mask) and its raw output (both rows). The kit's tests drive it.
    func forecastTrace(_ series: [Float], horizon: Int) async throws
        -> (forecast: QuantileForecast, inputs: [Float], raw: [Float])
    {
        let r = try await run(series, horizon: horizon, trace: true)
        return (r.forecast, r.inputs, r.raw)
    }

    private func run(_ series: [Float], horizon: Int, trace: Bool) async throws
        -> (forecast: QuantileForecast, inputs: [Float], raw: [Float])
    {
        let t0 = DispatchTime.now().uptimeNanoseconds
        guard !series.isEmpty else { throw KitTimestralForecasterError.emptySeries }
        guard horizon >= 1, horizon <= maximumHorizon else {
            throw KitTimestralForecasterError.horizonOutOfRange(requested: horizon, maximum: maximumHorizon)
        }
        let nPred = (horizon + TimestralHost.patch - 1) / TimestralHost.patch
        let patches = (series.count + TimestralHost.patch - 1) / TimestralHost.patch
        let fitting = graphs.filter { $0.bucket.horizonPatches >= nPred }
        // the smallest bucket that holds the whole series; else the one that reads the most of it
        guard let graph = fitting.first(where: { $0.bucket.contextPatches >= patches })
            ?? fitting.max(by: { $0.bucket.contextPatches < $1.bucket.contextPatches })
        else {
            throw KitTimestralForecasterError.horizonOutOfRange(requested: horizon, maximum: maximumHorizon)
        }
        let b = graph.bucket
        let context = TimestralHost.prepareContext(series, capacity: b.contextCapacity)
        guard context.observed > 0 else { throw KitTimestralForecasterError.noObservedValues(pointsRead: context.used) }

        let inputs = TimestralHost.graphInputs(context, nCtx: b.contextPatches, nHor: b.horizonPatches)
        let mirror = TimestralHost.mirrored(inputs.patches)
        let g0 = DispatchTime.now().uptimeNanoseconds
        let rows = try await graph.run(series: inputs.patches, mirror: mirror, cos: inputs.cos, sin: inputs.sin,
                                       mask: inputs.mask)
        let graphMs = Double(DispatchTime.now().uptimeNanoseconds - g0) / 1e6
        let per = b.maximumHorizon * TimestralHost.quantileCount
        let a = try TimestralHost.finishPass(rows[0..<per], context.x, context.mask, horizon: horizon)
        let negated = context.x.map { -$0 }
        let m = try TimestralHost.finishPass(rows[per..<(2 * per)], negated, context.mask, horizon: horizon)
        let flat = TimestralHost.combine(a, m, horizon: horizon, nonnegative: context.nonnegative)
        let n = TimestralHost.quantileCount
        let table = (0..<horizon).map { Array(flat[($0 * n)..<($0 * n + n)]) }
        let totalMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let forecast = QuantileForecast(
            levels: TimestralHost.levels, quantiles: table,
            details: .init(bucket: b, graphCalls: b.rows == 2 ? 1 : 2, contextUsed: context.used,
                           contextDropped: context.dropped, computeUnits: graph.computeUnits,
                           graphMilliseconds: graphMs, totalMilliseconds: totalMs))
        guard trace else { return (forecast, [], []) }
        return (forecast, inputs.patches + mirror + inputs.cos + inputs.sin + inputs.mask, rows)
    }
}

// MARK: - One graph

/// One loaded bucket graph and its contract: patches [R, L, 64], cos / sin [L+h, 64], attn_mask [1, 1, L+h, L+h]
/// in (one scalar type), quantiles [R, h·32, 21] out.
@available(macOS 27, iOS 27, *)
final class TimestralGraph: @unchecked Sendable {
    let url: URL
    let bucket: KitTimestralForecaster.Bucket
    let computeUnits: GraphModel.ComputeUnits
    private let model: AIModel  // kept alive for as long as the function is used
    private let function: InferenceFunction
    private let patchesDescriptor: NDArrayDescriptor
    private let tableDescriptor: NDArrayDescriptor
    private let maskDescriptor: NDArrayDescriptor

    /// fp16's largest finite value. An fp16 form clamps its output there, so a graph whose fp16 arithmetic overflowed
    /// returns it (the Neural Engine form on the iPhone's GPU did on every window of the zoo's fixture); a graph that
    /// computed right stays far below it (largest |q| 25.33 on that fixture, every form, Mac and iPhone).
    static let fp16Limit: Float = 65504

    /// The values at or beyond fp16's limit, ±infinity included. NaN is not counted: the host's own check
    /// (`degenerateInput`, the released forecaster's ValueError) reports it.
    static func saturatedCount(_ values: [Float]) -> Int {
        values.reduce(0) { $0 + ($1.magnitude >= fp16Limit ? 1 : 0) }
    }

    init(url: URL, computeUnits: GraphModel.ComputeUnits) async throws {
        self.url = url
        self.computeUnits = computeUnits
        let model = try await AIModel(contentsOf: url, options: computeUnits.specializationOptions)
        guard let function = try model.loadFunction(named: "main") else {
            throw KitTimestralForecasterError.bundle("\(url.lastPathComponent) has no function 'main'.")
        }
        let d = function.descriptor
        guard Set(d.inputNames) == ["patches", "cos", "sin", "attn_mask"], d.outputNames == ["quantiles"],
            d.stateNames.isEmpty,
            case .ndArray(let p) = d.inputDescriptor(of: "patches"),
            case .ndArray(let c) = d.inputDescriptor(of: "cos"),
            case .ndArray(let s) = d.inputDescriptor(of: "sin"),
            case .ndArray(let m) = d.inputDescriptor(of: "attn_mask"),
            case .ndArray(let q) = d.outputDescriptor(of: "quantiles")
        else {
            throw KitTimestralForecasterError.bundle(
                "\(url.lastPathComponent) is not a Timestral graph (inputs \(d.inputNames), outputs \(d.outputNames)).")
        }
        guard p.shape.count == 3, p.shape[2] == 64, p.shape[0] == 1 || p.shape[0] == 2, c.shape.count == 2,
            c.shape[1] == 64, s.shape == c.shape, m.shape == [1, 1, c.shape[0], c.shape[0]], c.shape[0] > p.shape[1],
            q.shape == [p.shape[0], (c.shape[0] - p.shape[1]) * TimestralHost.patch, TimestralHost.quantileCount],
            p.scalarType == c.scalarType, p.scalarType == s.scalarType, p.scalarType == m.scalarType,
            p.scalarType == .float32 || p.scalarType == .float16,
            q.scalarType == .float32 || q.scalarType == .float16
        else {
            throw KitTimestralForecasterError.bundle(
                "\(url.lastPathComponent): patches \(p.shape) cos \(c.shape) attn_mask \(m.shape) quantiles \(q.shape) "
                    + "is not Timestral's contract.")
        }
        let horizonPatches = c.shape[0] - p.shape[1]
        guard horizonPatches <= TimestralHost.blockPatches else {
            throw KitTimestralForecasterError.bundle(
                "\(url.lastPathComponent) forecasts \(horizonPatches) patches; one decode block is \(TimestralHost.blockPatches).")
        }
        bucket = .init(contextPatches: p.shape[1], horizonPatches: horizonPatches, rows: p.shape[0],
                       inputType: p.scalarType == .float16 ? "float16" : "float32")
        self.model = model
        self.function = function
        patchesDescriptor = p
        tableDescriptor = c
        maskDescriptor = m
    }

    /// Both rows through the graph — one call for R = 2, two for R = 1 — as [series row; mirror row] of
    /// (h·32, 21) float32 each. Throws `saturatedOutput` when the output reached fp16's limit.
    func run(series: [Float], mirror: [Float], cos: [Float], sin: [Float], mask: [Float]) async throws -> [Float] {
        let cosArray = try Self.make(cos, tableDescriptor)
        let sinArray = try Self.make(sin, tableDescriptor)
        let maskArray = try Self.make(mask, maskDescriptor)
        let feeds: [[Float]] = bucket.rows == 2 ? [series + mirror] : [series, mirror]
        var out: [Float] = []
        out.reserveCapacity(2 * bucket.maximumHorizon * TimestralHost.quantileCount)
        for rows in feeds {
            let patchesArray = try Self.make(rows, patchesDescriptor)
            var outputs = try await function.run(inputs: [
                "patches": patchesArray, "cos": cosArray, "sin": sinArray, "attn_mask": maskArray,
            ])
            guard let array = outputs.remove("quantiles")?.ndArray else {
                throw KitTimestralForecasterError.bundle("\(url.lastPathComponent) returned no quantiles.")
            }
            out += try Self.read(array)
        }
        let saturated = Self.saturatedCount(out)
        guard saturated == 0 else {
            throw KitTimestralForecasterError.saturatedOutput(
                graph: url.lastPathComponent, values: saturated, ranOn: KitTimestralForecaster.name(of: computeUnits))
        }
        return out
    }

    /// A new array of `descriptor` filled row-major from float32 values (rounded to nearest-even for a float16
    /// input, as NumPy's astype does).
    static func make(_ values: [Float], _ descriptor: NDArrayDescriptor) throws -> NDArray {
        var array = NDArray(descriptor: descriptor)
        switch descriptor.scalarType {
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: values)
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: values.map { Float16($0) })
        #endif
        default:
            throw KitTimestralForecasterError.bundle("input type \(descriptor.scalarType) is not float32 / float16")
        }
        return array
    }

    /// An output's scalars as float32, row-major, honoring its strides.
    static func read(_ array: NDArray) throws -> [Float] {
        switch array.scalarType {
        case .float32:
            return gather(array, as: Float.self).map { $0 }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            return gather(array, as: Float16.self).map { Float($0) }
        #endif
        default:
            throw KitTimestralForecasterError.bundle("output type \(array.scalarType) is not float32 / float16")
        }
    }

    private static func gather<T: BitwiseCopyable>(_ array: NDArray, as type: T.Type) -> [T] {
        let shape = array.shape
        let count = shape.reduce(1, *)
        return array.view(as: T.self).withUnsafePointer { ptr, _, strides in
            var expected = 1
            var contiguous = true
            for d in stride(from: shape.count - 1, through: 0, by: -1) {
                if shape[d] > 1 && strides[d] != expected {
                    contiguous = false
                    break
                }
                expected *= shape[d]
            }
            if contiguous { return Array(UnsafeBufferPointer(start: ptr, count: count)) }
            var out = [T]()
            out.reserveCapacity(count)
            var index = [Int](repeating: 0, count: shape.count)
            for _ in 0..<count {
                var offset = 0
                for d in 0..<shape.count { offset += index[d] * strides[d] }
                out.append(ptr[offset])
                var d = shape.count - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < shape[d] { break }
                    index[d] = 0
                    d -= 1
                }
            }
            return out
        }
    }
}
