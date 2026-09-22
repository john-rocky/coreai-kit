import CoreAIOps
import SwiftUI

/// The model drives: the road scrolls, rocks come, the car changes lane when its lane is
/// not clear. The prompt of the last tick sits behind a disclosure.
struct DriveView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = DriveModel()
    @State private var showingPrompt = false

    var body: some View {
        VStack(spacing: 10) {
            ScreenHeader(
                title: "Drive",
                subtitle: "Every tick the model reads the lanes ahead and picks one — a game state in, an action out")
            controlsLayout {
                HStack(spacing: 10) {
                    Button(model.running ? "Stop" : (model.crashed ? "Drive again" : "Drive")) {
                        if model.running { model.stop() } else { model.start(runtime) }
                    }
                    .buttonStyle(.borderedProminent)
                    Button(showingPrompt ? "Hide the prompt" : "How it decides") { showingPrompt.toggle() }
                }
                HStack {
                    Spacer()
                    Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
            }
            HStack(alignment: .top, spacing: 16) {
                road
                    .frame(maxWidth: isPhone ? .infinity : 360)
                    .frame(maxHeight: .infinity)
                if showingPrompt && !isPhone { prompt.frame(maxWidth: .infinity) }
            }
            if showingPrompt && isPhone { prompt }
        }
        .padding()
        .task {
            await autoplay.run(.drive, runtime: runtime, status: { model.status }) {
                showingPrompt = true
                model.start(runtime)
                let end = Date().addingTimeInterval(25)
                while Date() < end, model.running { try? await Task.sleep(for: .milliseconds(200)) }
                model.stop()
            }
        }
    }

    private var road: some View {
        GeometryReader { geo in
            let rowCount = DriveModel.rowsAhead + 1
            let cellHeight = geo.size.height / CGFloat(rowCount)
            let laneWidth = geo.size.width / CGFloat(DriveModel.lanes)
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 12).fill(Color(white: 0.22))
                // lane markings
                ForEach(1..<DriveModel.lanes, id: \.self) { l in
                    Rectangle().fill(.white.opacity(0.35))
                        .frame(width: 2, height: geo.size.height)
                        .offset(x: laneWidth * CGFloat(l) - 1)
                }
                // rocks: rows[0] at the bottom, far rows at the top
                ForEach(Array(model.rows.enumerated()), id: \.offset) { index, row in
                    ForEach(0..<DriveModel.lanes, id: \.self) { l in
                        if row[l] == .rock {
                            Text("🪨").font(.system(size: cellHeight * 0.6))
                                .frame(width: laneWidth, height: cellHeight)
                                .offset(x: laneWidth * CGFloat(l), y: cellHeight * CGFloat(rowCount - 1 - index))
                        }
                    }
                }
                Text(model.crashed ? "💥" : "🚗").font(.system(size: cellHeight * 0.7))
                    .frame(width: laneWidth, height: cellHeight)
                    .offset(x: laneWidth * CGFloat(model.lane), y: cellHeight * CGFloat(rowCount - 1))
                    .animation(.easeInOut(duration: 0.18), value: model.lane)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .aspectRatio(CGFloat(DriveModel.lanes) / CGFloat(DriveModel.rowsAhead + 1) * 1.6, contentMode: .fit)
    }

    private var prompt: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("The last tick's prompt").font(.caption.bold()).foregroundStyle(.secondary)
            Text(model.lastState).font(.callout)
            Text(DriveModel.question).font(.callout).foregroundStyle(.secondary)
            ForEach(model.seen) { option in
                HStack {
                    Text(option.text).font(.callout.monospaced())
                    Spacer()
                    ProbabilityBar(value: option.probability, tint: option.lane == model.lane ? .green : .gray)
                        .frame(width: 90)
                    Text(option.probability.formatted(.number.precision(.fractionLength(2))))
                        .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                }
            }
            if !model.milliseconds.isEmpty {
                Text("\(model.decisions) decisions · median \(ms(model.medianMilliseconds)) · last \(ms(model.milliseconds.last ?? 0))")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.4)))
    }
}
