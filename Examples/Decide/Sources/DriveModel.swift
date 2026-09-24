// DriveModel — screen: a car on a three-lane road, and the model drives it. Every tick the
// road advances one row; the model reads the lanes it can reach — "left lane: rock 1 ahead",
// "middle lane: clear" — and picks the lane as a choice. Nothing is generated: one scored
// prompt per tick, and the tick rate is the decision rate. It is the shape the most-viewed
// System One posts use (a driving simulator, Mario, Minecraft, Tetris): the game state as
// text, the next action as a typed choice.
//
// The shape decides whether it works. Asked "which move?" with "stay in lane / move left /
// move right", MiniCPM5 2B stays in its lane even into a rock (2026-09-23, 4 crashes in 12
// states). Asked "which lane?" with each reachable lane described by what lies ahead in it,
// the same model never picks a rock lane (12/12) and keeps its lane while it is clear.

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class DriveModel {
    enum Cell: Equatable { case clear, rock }

    /// What the model saw and answered on the last tick.
    struct Seen: Identifiable {
        let lane: Int
        let text: String
        let probability: Double
        var id: Int { lane }
    }

    nonisolated static let lanes = 3
    nonisolated static let laneNames = ["left", "middle", "right"]
    /// Rows of road drawn ahead of the car.
    nonisolated static let rowsAhead = 8
    /// Rows the lane descriptions look ahead: a decision needs the near future, not the horizon.
    /// Two, and the road never puts rocks in different lanes on consecutive rows, so at most one
    /// reachable lane is ever blocked and a "clear" option always exists — the comparison the
    /// model gets right. Offered "rock 1 ahead" against "rock 2 ahead" it takes the nearer rock
    /// (2026-09-23, a crash at row 30 of a recording).
    static let lookahead = 2
    /// Seconds per tick at least, so a fast Mac stays watchable; the decision is timed separately.
    static let minimumTick = 0.3
    /// Chance that a new row carries a rock (never more than one per row, so a clear lane is
    /// always within one move).
    static let rockChance = 0.55
    nonisolated static let question = "Which lane should the car drive in next? A rock in a lane crashes the car; clear is safe."

    /// `rows[0]` is the row the car is on; `rows[1...]` lie ahead, nearest first.
    private(set) var rows: [[Cell]] = []
    private(set) var lane = 1
    private(set) var running = false
    private(set) var crashed = false
    private(set) var distance = 0
    private(set) var decisions = 0
    private(set) var milliseconds: [Double] = []
    private(set) var dodged = 0
    private(set) var lastState = ""
    private(set) var seen: [Seen] = []
    var status = "Press Drive — every tick the model reads the lanes ahead and picks one."
    private var task: Task<Void, Never>?

    var medianMilliseconds: Double { median(milliseconds) }

    init() { reset() }

    func reset() {
        rows = (0...Self.rowsAhead).map { _ in Array(repeating: Cell.clear, count: Self.lanes) }
        lane = 1
        crashed = false
        distance = 0
        decisions = 0
        milliseconds = []
        dodged = 0
        lastState = ""
        seen = []
    }

    func start(_ runtime: DecideRuntime) {
        guard !running else { return }
        if crashed { reset() }
        running = true
        status = "Loading…"
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let decider = try await runtime.ready()
                while self.running, !self.crashed, !Task.isCancelled {
                    let tickStart = Date()
                    try await self.tick(decider)
                    let elapsed = Date().timeIntervalSince(tickStart)
                    if elapsed < Self.minimumTick {
                        try? await Task.sleep(for: .seconds(Self.minimumTick - elapsed))
                    }
                }
            } catch {
                self.status = "Error: \(error.localizedDescription)"
            }
            self.running = false
        }
    }

    func stop() {
        running = false
        task?.cancel()
        task = nil
        if !crashed {
            status = "Stopped after \(distance) rows · \(decisions) decisions · median \(ms(medianMilliseconds)) each · \(dodged) rocks passed"
        }
    }

    /// One tick: describe the reachable lanes, let the model pick one, move, advance the road.
    private func tick(_ decider: TypedDecisions) async throws {
        let reachable = Array(max(0, lane - 1)...min(Self.lanes - 1, lane + 1))
        let options = reachable.map { l in
            Decision.Option(id: Self.laneNames[l], description: "\(Self.laneNames[l]) lane: \(describe(lane: l))")
        }
        let state = "The car is in the \(Self.laneNames[lane]) lane; it can stay or move to a neighbouring lane."
        let answer = try await decider.decide(state, .choice(Self.question, options: options))
        decisions += 1
        milliseconds.append(answer.timing.milliseconds)
        lastState = state
        if case .choice(let c) = answer.value {
            seen = zip(reachable, options).map { l, option in
                Seen(lane: l, text: option.description, probability: c.probabilities[option.id] ?? 0)
            }
            if let index = Self.laneNames.firstIndex(of: c.id) { lane = index }
        }
        // The road advances one row: the car is now on what was the nearest row ahead.
        let rockWasAhead = rows[1].contains(.rock)
        rows.removeFirst()
        rows.append(newRow())
        distance += 1
        if rows[0][lane] == .rock {
            crashed = true
            running = false
            status = "Crashed after \(distance) rows · \(decisions) decisions · median \(ms(medianMilliseconds)) each"
        } else {
            if rockWasAhead { dodged += 1 }
            status = "\(distance) rows · \(decisions) decisions · median \(ms(medianMilliseconds)) each · \(dodged) rocks passed"
        }
    }

    /// The nearest rock in `lane` within the lookahead, as the model reads it.
    private func describe(lane l: Int) -> String {
        for d in 1...Self.lookahead where d < rows.count {
            if rows[d][l] == .rock { return "rock \(d) ahead" }
        }
        return "clear"
    }

    /// A new far row: at most one rock, and a rock only where the previous row has none or has
    /// its rock in the same lane — so within any two rows at most one lane is blocked.
    private func newRow() -> [Cell] {
        var row = Array(repeating: Cell.clear, count: Self.lanes)
        guard Double.random(in: 0..<1) < Self.rockChance else { return row }
        let previous = rows.last ?? row
        if let blocked = previous.firstIndex(of: .rock) {
            if Bool.random() { row[blocked] = .rock }
        } else {
            row[Int.random(in: 0..<Self.lanes)] = .rock
        }
        return row
    }
}
