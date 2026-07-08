import Foundation

// forecast-cli — headless TimesFM 2.5 forecasting.
//
//   swift run forecast-cli --csv series.csv            # one number per line (or comma-separated)
//   swift run forecast-cli --demo                      # a built-in seasonal series
//   swift run forecast-cli --csv s.csv --horizon 24    # print only the first 24 steps
//
// Prints the point forecast plus the 10th/90th-percentile band (prediction interval).

func parseArgs() -> (csv: String?, demo: Bool, horizon: Int, model: String) {
    var csv: String? = nil, demo = false, horizon = 128, model = "timesfm-2.5-200m"
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--csv": csv = it.next()
        case "--demo": demo = true
        case "--horizon": horizon = Int(it.next() ?? "") ?? 128
        case "--model": model = it.next() ?? model
        default: FileHandle.standardError.write(Data("unknown arg: \(a)\n".utf8))
        }
    }
    return (csv, demo, horizon, model)
}

func loadSeries(csv path: String) throws -> [Float] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text
        .split(whereSeparator: { $0 == "\n" || $0 == "," || $0 == "\r" })
        .compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
}

func demoSeries() -> [Float] {
    (0..<512).map { i in 10 + 3 * sinf(2 * .pi * Float(i) / 48) + 0.01 * Float(i) }
}

let args = parseArgs()
let series: [Float]
if let csv = args.csv {
    series = try loadSeries(csv: csv)
} else if args.demo {
    series = demoSeries()
} else {
    FileHandle.standardError.write(Data("usage: forecast-cli --csv <file> | --demo\n".utf8))
    exit(2)
}
guard !series.isEmpty else {
    FileHandle.standardError.write(Data("no numeric values found\n".utf8))
    exit(1)
}

let t0 = Date()
let f = try await forecast(series, model: args.model)
let dt = Date().timeIntervalSince(t0)

let n = min(args.horizon, f.mean.count)
print("context: \(series.count) points  →  \(n)-step forecast  (\(String(format: "%.0f", dt * 1000)) ms)")
print("step   p10        mean       p90")
for h in 0..<n {
    let p10 = f.quantiles[h][1]                 // decile 0.1
    let p90 = f.quantiles[h][9]                 // decile 0.9
    print(String(format: "%4d  %9.3f  %9.3f  %9.3f", h + 1, p10, f.mean[h], p90))
}
