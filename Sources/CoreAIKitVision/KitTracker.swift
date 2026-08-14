// KitTracker.swift — the thing every detection app needs second.
//
// A detector answers "what is in this frame". The question directly after it is always "is
// that the same one as last frame" — count the people, time how long the car waited, follow
// the ball. Apple ships no answer, the models do not either (a detector is stateless by
// construction), and the algorithm is the same every time. So it is written once, here, with
// no model and no framework in it.
//
// ```swift
// let tracker = KitTracker()
// for try await result in watch {
//     let tracks = tracker.update(result.value, at: CACurrentMediaTime())
//     for t in tracks where t.isConfirmed { draw(t.box, id: t.id) }
// }
// ```
//
// WHAT IS AND IS NOT IN HERE
// --------------------------
// Association is Hungarian — optimal, not greedy. Greedy IoU matching is twenty lines and
// gets the easy frames right; it swaps identities exactly when two objects pass each other,
// which is the frame anybody looks at. Doing that properly is most of the value.
//
// Motion is constant velocity with variable dt, not a Kalman filter. That is a deliberate
// trade and worth stating: a Kalman filter buys uncertainty-aware gating, which matters in a
// crowd, and costs a noise model that has to be tuned per camera and per frame rate to not be
// worse than nothing. This pipeline drops frames by design, so dt is not constant and a
// filter tuned at 30 fps is being fed 8 fps intervals whenever the device gets hot. Constant
// velocity has one parameter, degrades honestly, and takes dt as an argument.
//
// Association runs in two passes over confidence, the one idea from ByteTrack worth having:
// a partly-occluded object does not vanish, its score drops. Throwing away every detection
// under the display threshold throws away exactly the frames where identity is hard to keep.

import CoreGraphics
import Foundation

/// One object, followed across frames.
public struct Track: Identifiable, Sendable {
    /// Stable for the life of the object. Never reused within a tracker.
    public let id: Int
    public let classID: Int
    public let label: String
    /// Normalized, origin top-left — the same space `Detection.box` is in. While the object
    /// is unmatched this is the predicted position, not a measurement.
    public var box: CGRect
    public var score: Float
    /// Frames this track has been matched to a detection.
    public var hits: Int
    /// Frames since the last match. Zero means this box is a measurement.
    public var timeSinceUpdate: Int
    /// True once the track has survived `minHits`. An unconfirmed track is a candidate, not
    /// an object — drawing them is how a detector's flicker becomes the tracker's fault.
    public var isConfirmed: Bool
    /// Recent boxes, newest last, for a trail or a direction. Capped by `Options.historyLimit`.
    public var history: [CGRect]

    /// Normalized velocity per second, from the last two measurements. Zero until a track has
    /// been matched twice.
    public var velocity: CGVector
}

/// Follows detections across frames and gives them stable identities.
///
/// Not an actor and not `Sendable`: it is a state machine over consecutive frames and its
/// whole contract is that calls are ordered. Handing it to two tasks would interleave frames
/// and produce identity switches that look like tracker bugs. Own it from one place — in the
/// live pipeline that is the loop consuming the stream.
public final class KitTracker {
    public struct Options: Sendable {
        /// Minimum box overlap to consider a track and a detection the same object.
        public var iouThreshold: Float
        /// Detections at or above this score take part in the first association pass.
        public var highScore: Float
        /// Detections below this are ignored entirely.
        public var lowScore: Float
        /// Frames a track survives unmatched before it is dropped. At 15 fps, 30 is two
        /// seconds of occlusion.
        public var maxAge: Int
        /// Matches before a track is confirmed. 1 shows everything immediately and inherits
        /// every detector false positive.
        public var minHits: Int
        /// Whether a track may match a detection of a different class. Off by default: a
        /// person does not become a dog, and a detector that flickers between two labels
        /// should not be papered over here.
        public var matchAcrossClasses: Bool
        /// Boxes kept in `Track.history`.
        public var historyLimit: Int
        /// How far an object may plausibly travel in one second, in normalized units, for
        /// the distance fallback below. Zero disables the fallback and leaves association
        /// on IoU alone.
        public var maxTravelPerSecond: Double

        public init(
            iouThreshold: Float = 0.3, highScore: Float = 0.5, lowScore: Float = 0.1,
            maxAge: Int = 30, minHits: Int = 3, matchAcrossClasses: Bool = false,
            historyLimit: Int = 30, maxTravelPerSecond: Double = 1.0
        ) {
            self.maxTravelPerSecond = maxTravelPerSecond
            self.iouThreshold = iouThreshold
            self.highScore = highScore
            self.lowScore = lowScore
            self.maxAge = maxAge
            self.minHits = minHits
            self.matchAcrossClasses = matchAcrossClasses
            self.historyLimit = historyLimit
        }
    }

    private var options: Options
    private var tracks: [Track] = []
    private var lastTime: TimeInterval?
    private var nextID = 1

    public init(options: Options = Options()) {
        self.options = options
    }

    /// Everything currently tracked, including unconfirmed candidates and tracks coasting
    /// through an occlusion. Filter on `isConfirmed` to draw.
    public var current: [Track] { tracks }

    public func reset() {
        tracks.removeAll()
        lastTime = nil
        // Ids are not reused: a chart keyed by track id must not silently merge two objects
        // across a reset.
    }

    /// One frame. `time` is seconds on any monotonic clock — `CACurrentMediaTime()` live,
    /// `VideoFrame.time` for a file. It must be a real timestamp rather than a frame counter
    /// because the live pipeline drops frames, so the interval between two calls varies with
    /// how busy the model and the device are.
    @discardableResult
    public func update(_ detections: [Detection], at time: TimeInterval) -> [Track] {
        let dt = lastTime.map { max(time - $0, 0) } ?? 0
        lastTime = time

        // 1. Predict. An unmatched track coasts on its last velocity so that association has
        //    something plausible to match against rather than a stale box.
        for i in tracks.indices {
            tracks[i].timeSinceUpdate += 1
            if dt > 0, tracks[i].velocity != .zero {
                tracks[i].box.origin.x += tracks[i].velocity.dx * dt
                tracks[i].box.origin.y += tracks[i].velocity.dy * dt
            }
        }

        let usable = detections.filter { $0.score >= options.lowScore }
        let high = usable.filter { $0.score >= options.highScore }
        let low = usable.filter { $0.score < options.highScore }

        // 2. Confident detections against every track.
        var unmatchedTracks = Set(tracks.indices)
        var matched = associate(trackIndices: unmatchedTracks, detections: high, dt: dt)
        for m in matched {
            apply(high[m.detection], to: m.track, dt: dt)
            unmatchedTracks.remove(m.track)
        }
        let unmatchedHigh = Set(high.indices).subtracting(matched.map(\.detection))

        // 3. Faint detections against what is left. This is the occlusion recovery: a person
        //    behind a pillar is still detected, just weakly, and matching that keeps the id.
        //    Only already-established tracks take part — a faint blob must not create one.
        let established = unmatchedTracks.filter { tracks[$0].hits >= options.minHits }
        matched = associate(trackIndices: Set(established), detections: low, dt: dt)
        for m in matched {
            apply(low[m.detection], to: m.track, dt: dt)
            unmatchedTracks.remove(m.track)
        }

        // 4. Births, from confident detections only.
        for d in unmatchedHigh.sorted() {
            tracks.append(makeTrack(high[d]))
        }

        // 5. Deaths. An unconfirmed track dies the moment it misses, because it was never
        //    established as an object; a confirmed one is allowed to coast.
        tracks.removeAll { track in
            track.timeSinceUpdate > (track.isConfirmed ? options.maxAge : 0)
        }
        return tracks
    }

    // MARK: - Association

    private struct Match {
        let track: Int
        let detection: Int
    }

    /// Optimal assignment between the given tracks and detections, subject to the IoU floor
    /// (or the travel gate) and the class rule.
    ///
    /// **IoU alone is not enough here, and the reason is this package's own thermal
    /// governor.** Overlap between consecutive frames falls with the interval: a 0.1-wide
    /// box moving 0.3 units per second overlaps 0.82 at 30 fps and 0.25 at 5 fps — below any
    /// sane threshold. So an IoU-only tracker works on a cool phone and quietly assigns a new
    /// id to every object on a hot one, which is the worst possible failure to debug. Boxes
    /// that do not overlap therefore fall back to centre distance, gated by how far the
    /// object could actually have travelled in the elapsed time. IoU matches always outrank
    /// distance matches, because overlap is the stronger evidence.
    private func associate(
        trackIndices: Set<Int>, detections: [Detection], dt: TimeInterval
    ) -> [Match] {
        let ts = trackIndices.sorted()
        guard !ts.isEmpty, !detections.isEmpty else { return [] }

        // Cost is 1 - IoU, with an impossible pair pushed above any real cost so the solver
        // never chooses it and the floor check below rejects it if it somehow does.
        let impossible: Float = 3
        // Travel budget for this interval. Without a previous frame (dt == 0) there is no
        // budget and only overlap can match, which is correct: a first frame has no motion.
        let budget = options.maxTravelPerSecond * dt
        var cost = [[Float]](
            repeating: [Float](repeating: impossible, count: detections.count),
            count: ts.count)
        for (row, t) in ts.enumerated() {
            for (col, d) in detections.enumerated() {
                guard options.matchAcrossClasses || tracks[t].classID == d.classID else {
                    continue
                }
                let overlap = KitTracker.iou(tracks[t].box, d.box)
                if overlap >= options.iouThreshold {
                    cost[row][col] = 1 - overlap  // 0…1: always preferred
                    continue
                }
                guard budget > 0 else { continue }
                let dx = d.box.midX - tracks[t].box.midX
                let dy = d.box.midY - tracks[t].box.midY
                let distance = (dx * dx + dy * dy).squareRoot()
                // Half a box of slack on top of the travel budget: the detector's own box
                // jitter is on that scale and is not motion.
                let reach = budget + Double(max(d.box.width, d.box.height)) / 2
                if distance <= reach {
                    cost[row][col] = 1 + Float(distance / reach)  // 1…2: the fallback band
                }
            }
        }

        return hungarian(cost).compactMap { row, col in
            cost[row][col] < impossible ? Match(track: ts[row], detection: col) : nil
        }
    }

    private func apply(_ detection: Detection, to index: Int, dt: TimeInterval) {
        var track = tracks[index]
        if dt > 0, track.timeSinceUpdate > 0 || track.hits > 0 {
            // Velocity from where it was measured to where it is now. Smoothed, because a
            // one-frame jump from a jittery box otherwise throws the prediction across the
            // frame during the next occlusion.
            let observed = CGVector(
                dx: (detection.box.midX - track.box.midX) / dt,
                dy: (detection.box.midY - track.box.midY) / dt)
            track.velocity = track.hits <= 1
                ? observed
                : CGVector(dx: track.velocity.dx * 0.6 + observed.dx * 0.4,
                           dy: track.velocity.dy * 0.6 + observed.dy * 0.4)
        }
        track.box = detection.box
        track.score = detection.score
        track.hits += 1
        track.timeSinceUpdate = 0
        track.isConfirmed = track.isConfirmed || track.hits >= options.minHits
        track.history.append(detection.box)
        if track.history.count > options.historyLimit { track.history.removeFirst() }
        tracks[index] = track
    }

    private func makeTrack(_ detection: Detection) -> Track {
        defer { nextID += 1 }
        return Track(
            id: nextID, classID: detection.classID, label: detection.label,
            box: detection.box, score: detection.score, hits: 1, timeSinceUpdate: 0,
            isConfirmed: options.minHits <= 1, history: [detection.box], velocity: .zero)
    }

    /// Intersection over union of two normalized boxes.
    static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let overlap = a.intersection(b)
        guard !overlap.isNull, overlap.width > 0, overlap.height > 0 else { return 0 }
        let intersection = overlap.width * overlap.height
        let union = a.width * a.height + b.width * b.height - intersection
        return union > 0 ? Float(intersection / union) : 0
    }
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// Minimum-cost assignment (Hungarian / Kuhn–Munkres, O(n²m)).
///
/// Greedy nearest-match is the obvious alternative and it is wrong in the one frame that
/// matters: when two objects cross, greedy takes the best single pair first and forces the
/// other into the leftover, swapping their identities. That is the frame a person notices,
/// and an optimal assignment over the whole cost matrix is what avoids it.
///
/// The implementation is the standard potentials/augmenting-path form, which handles a
/// rectangular matrix directly — the row and column counts here are tracks and detections and
/// are rarely equal.
func hungarian(_ cost: [[Float]]) -> [(row: Int, col: Int)] {
    let rows = cost.count
    guard rows > 0 else { return [] }
    let realCols = cost[0].count
    guard realCols > 0 else { return [] }

    // The augmenting-path form below needs at least as many columns as rows: with fewer, the
    // search for a free column runs off the end. More tracks than detections is not an edge
    // case here — it is what every frame looks like when an object leaves — so the matrix is
    // padded out with a cost no real pair can reach, and the padded columns are dropped from
    // the result. (Found by a test, after the first version crashed on three tracks and one
    // detection.)
    let cols = max(rows, realCols)
    // One more than the largest real cost, not a huge sentinel: the solver subtracts
    // accumulated potentials from these values, and a near-infinite pad overflows into inf
    // and then nan, at which point every comparison is false and the search never
    // terminates. Big enough to never be preferred is the whole requirement.
    let padding = (cost.flatMap { $0 }.max() ?? 0) + 1
    let cost = realCols == cols
        ? cost
        : cost.map { $0 + [Float](repeating: padding, count: cols - realCols) }

    // Potentials, and the column -> row assignment being built. Index 0 is a sentinel, which
    // is why every array is one longer than it looks like it should be.
    var u = [Float](repeating: 0, count: rows + 1)
    var v = [Float](repeating: 0, count: cols + 1)
    var assignedRow = [Int](repeating: 0, count: cols + 1)
    var way = [Int](repeating: 0, count: cols + 1)

    for row in 1...rows {
        assignedRow[0] = row
        var col0 = 0
        var minimum = [Float](repeating: .greatestFiniteMagnitude, count: cols + 1)
        var used = [Bool](repeating: false, count: cols + 1)
        repeat {
            used[col0] = true
            let currentRow = assignedRow[col0]
            var delta = Float.greatestFiniteMagnitude
            var nextCol = 0
            for col in 1...cols where !used[col] {
                let candidate = cost[currentRow - 1][col - 1] - u[currentRow] - v[col]
                if candidate < minimum[col] {
                    minimum[col] = candidate
                    way[col] = col0
                }
                if minimum[col] < delta {
                    delta = minimum[col]
                    nextCol = col
                }
            }
            for col in 0...cols {
                if used[col] {
                    u[assignedRow[col]] += delta
                    v[col] -= delta
                } else {
                    minimum[col] -= delta
                }
            }
            col0 = nextCol
        } while assignedRow[col0] != 0
        // Walk the augmenting path back, flipping the assignment as we go.
        repeat {
            let previous = way[col0]
            assignedRow[col0] = assignedRow[previous]
            col0 = previous
        } while col0 != 0
    }

    return (1...cols)
        .filter { assignedRow[$0] != 0 && $0 <= realCols }
        .map { (row: assignedRow[$0] - 1, col: $0 - 1) }
}
