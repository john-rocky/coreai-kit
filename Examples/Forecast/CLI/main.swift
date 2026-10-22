import CoreAIKit
import Foundation

// forecast-cli — headless forecasting with TimesFM 2.5 or Timestral 209M.
//
//   swift run forecast-cli --demo                                        # TimesFM 2.5, a built-in seasonal series
//   swift run forecast-cli --series sales.csv                            # your series: a number per line, or a CSV
//   swift run forecast-cli --model timestral-209m --series sales.csv --horizon 48 --band 90
//   swift run forecast-cli --model timestral-209m --bundle <folder or graph> --series sales.csv
//   pbpaste | swift run forecast-cli --model timestral-209m --series -   # pasted numbers on stdin
//
// --horizon N   steps to forecast (TimesFM: 1…128, default 128; Timestral: 1…1,024, default 128)
// --band 80|90|98  the interval printed beside the median (Timestral; TimesFM has 80: p10–p90)
// --compute auto|gpu|ane  where Timestral's graphs run: auto (default) = each on its form's unit (the Neural Engine
//               form on the Neural Engine, the others on the GPU); gpu / ane = that unit for every graph, which the
//               Neural Engine form refuses for gpu
// --repeat N    forecast N times with one loaded model and print each time (the opening one is the cold one)
// Prints the median and the band per step, then what ran. A bad input or horizon prints the reason, exit 1.

struct Options {
    var series: String?
    var demo = false
    var horizon: Int?
    var model = "timesfm-2.5-200m"
    var bundle: String?
    var band = 80
    var compute = "auto"
    var repeatCount = 1
}

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(code)
}

func parseArgs() -> Options {
    var o = Options()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    func value(_ flag: String) -> String {
        guard let v = it.next() else { fail("\(flag) needs a value", code: 2) }
        return v
    }
    func whole(_ flag: String) -> Int {
        let v = value(flag)
        guard let n = Int(v) else { fail("\(flag) takes a whole number, not '\(v)'", code: 2) }
        return n
    }
    while let a = it.next() {
        switch a {
        case "--series", "--csv": o.series = value(a)
        case "--demo": o.demo = true
        case "--horizon": o.horizon = whole(a)
        case "--model": o.model = value(a)
        case "--bundle": o.bundle = value(a)
        case "--band": o.band = whole(a)
        case "--compute":
            o.compute = value(a)
            guard ["auto", "gpu", "ane"].contains(o.compute) else {
                fail("--compute is auto, gpu or ane, not '\(o.compute)'", code: 2)
            }
        case "--repeat": o.repeatCount = max(1, whole(a))
        case "--help", "-h":
            print("usage: forecast-cli (--series <file> | --series - | --demo) [--model timesfm-2.5-200m|timestral-209m]")
            print("       [--bundle <folder or graph>] [--horizon N] [--band 80|90|98] [--compute auto|gpu|ane] [--repeat N]")
            exit(0)
        default: fail("unknown argument '\(a)' (see --help)", code: 2)
        }
    }
    return o
}

func demoSeries() -> [Float] {
    (0..<512).map { i in 10 + 3 * sinf(2 * .pi * Float(i) / 48) + 0.01 * Float(i) }
}

func readSeries(_ o: Options) -> [Float] {
    if o.demo { return demoSeries() }
    guard let path = o.series else { fail("give a series: --series <file>, --series - (stdin) or --demo", code: 2) }
    let text: String
    if path == "-" {
        text = String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
    } else {
        do { text = try String(contentsOfFile: path, encoding: .utf8) } catch {
            fail("cannot read \(path): \(error.localizedDescription)")
        }
    }
    do {
        let input = try SeriesParser.parse(text)
        let column = input.column.map { ", column '\($0)'" } ?? ""
        print("series: \(input.values.count) values (\(input.missing) missing)\(column)")
        return input.values
    } catch {
        fail(error.localizedDescription)
    }
}

func row(_ step: Int, _ lo: Float, _ mid: Float, _ hi: Float) -> String {
    String(format: "%4d  %11.4f  %11.4f  %11.4f", step, lo, mid, hi)
}

func header(_ lo: String, _ hi: String) -> String {
    func right(_ s: String) -> String { String(repeating: " ", count: max(0, 11 - s.count)) + s }
    return "step  " + right(lo) + "  " + right("median") + "  " + right(hi)
}

/// Where a forecast's graph ran, for the summary line.
func unitName(_ d: QuantileForecast.Details) -> String {
    switch d.computeUnits {
    case .neuralEngine: return "Neural Engine"
    case .gpu: return "GPU"
    default: return "CPU"
    }
}

let o = parseArgs()
let series = readSeries(o)

switch o.model {
case "timesfm-2.5-200m":
    let h = o.horizon ?? 128
    guard (1...128).contains(h) else { fail("TimesFM 2.5 forecasts 1…128 steps; --horizon \(h) is outside") }
    guard !series.isEmpty else { fail("The series is empty: give at least one number.") }
    let t0 = Date()
    let f: Forecast
    do { f = try await forecast(series, model: o.model) } catch { fail(error.localizedDescription) }
    let ms = Date().timeIntervalSince(t0) * 1000
    print("model: TimesFM 2.5 200M · \(h)-step forecast · \(String(format: "%.0f", ms)) ms (load included)")
    print(header("p10", "p90"))
    for t in 0..<h { print(row(t + 1, f.quantiles[t][1], f.mean[t], f.quantiles[t][9])) }

case KitTimestralForecaster.catalogID:
    let h = o.horizon ?? 128
    let (lo, hi): (Float, Float) = {
        switch o.band {
        case 80: return (0.1, 0.9)
        case 90: return (0.05, 0.95)
        case 98: return (0.01, 0.99)
        default: fail("--band is 80, 90 or 98, not \(o.band)", code: 2)
        }
    }()
    let forecaster: KitTimestralForecaster
    let tLoad = Date()
    do {
        let bundle = o.bundle.map { URL(fileURLWithPath: $0) }
        if o.compute == "auto" {
            forecaster = try await loadTimestral(bundle: bundle)
        } else if let bundle {
            forecaster = try await KitTimestralForecaster(
                bundleAt: bundle, computeUnits: o.compute == "ane" ? .neuralEngine : .gpu)
        } else {
            forecaster = try await KitTimestralForecaster(computeUnits: o.compute == "ane" ? .neuralEngine : .gpu)
        }
    } catch CoreAIKitError.modelNotInCatalog {
        fail("\(KitTimestralForecaster.catalogID) is not in this kit's catalog yet: pass --bundle <folder or graph>")
    } catch {
        fail(error.localizedDescription)
    }
    let loadS = Date().timeIntervalSince(tLoad)
    var last: QuantileForecast?
    for i in 1...o.repeatCount {
        do {
            let q = try await forecaster.forecast(series, horizon: h)
            if o.repeatCount > 1 {
                print(String(format: "forecast %ld: %.2f ms (graph %.2f ms)", i, q.details.totalMilliseconds,
                             q.details.graphMilliseconds))
            }
            last = q
        } catch {
            fail(error.localizedDescription)
        }
    }
    guard let q = last else { exit(1) }
    let d = q.details
    let dropped = d.contextDropped > 0 ? " (the oldest \(d.contextDropped) left out)" : ""
    print("model: Timestral 209M · bucket \(d.bucket) on the \(unitName(d)) · read \(d.contextUsed) points\(dropped) "
          + "· \(h)-step forecast")
    print(String(format: "time: load %.2f s · forecast %.2f ms (graph %.2f ms, %ld call%@)", loadS, d.totalMilliseconds,
                 d.graphMilliseconds, d.graphCalls, d.graphCalls == 1 ? "" : "s"))
    let pl = q.quantile(lo)!, med = q.median, ph = q.quantile(hi)!
    print(header("p\(Int((lo * 100).rounded()))", "p\(Int((hi * 100).rounded()))"))
    for t in 0..<h { print(row(t + 1, pl[t], med[t], ph[t])) }

default:
    fail("--model is timesfm-2.5-200m or \(KitTimestralForecaster.catalogID), not '\(o.model)'", code: 2)
}
