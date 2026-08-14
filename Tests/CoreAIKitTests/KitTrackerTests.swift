// KitTrackerTests.swift — identity, which is the only thing a tracker is for.
//
// A tracker that keeps ids on easy frames is worth nothing; every twenty-line greedy matcher
// does that. What has to hold is the hard frame: two objects crossing, one disappearing
// behind something, a detector that flickers. Those are all constructible from arrays of
// rectangles, so none of this needs a model, a camera, or a device.

import CoreGraphics
import Foundation
import Testing

@testable import CoreAIKitVision

struct HungarianTests {
    /// Greedy takes the global minimum (1, at row 0) and is then forced into 100, totalling
    /// 101. The optimal assignment is 2 + 2 = 4. This is the crossing frame, in miniature.
    @Test func optimalBeatsGreedy() {
        let cost: [[Float]] = [
            [1, 2],
            [2, 100],
        ]
        let pairs = hungarian(cost).sorted { $0.row < $1.row }
        #expect(pairs.map(\.col) == [1, 0])
        #expect(pairs.reduce(Float(0)) { $0 + cost[$1.row][$1.col] } == 4)
    }

    @Test func handlesMoreDetectionsThanTracks() {
        let pairs = hungarian([[5, 1, 9]])
        #expect(pairs.count == 1)
        #expect(pairs[0] == (row: 0, col: 1))
    }

    @Test func handlesMoreTracksThanDetections() {
        let pairs = hungarian([[5], [1], [9]])
        #expect(pairs.count == 1)
        #expect(pairs[0] == (row: 1, col: 0))
    }

    @Test func emptyInputIsEmptyOutput() {
        #expect(hungarian([]).isEmpty)
        #expect(hungarian([[]]).isEmpty)
    }

    /// Every row assigned at most once, every column at most once — the property the whole
    /// algorithm exists to guarantee, checked on a matrix with no obvious structure.
    @Test func theAssignmentIsAPermutation() {
        let cost: [[Float]] = [
            [4, 1, 3, 9],
            [2, 0, 5, 7],
            [3, 2, 2, 1],
        ]
        let pairs = hungarian(cost)
        #expect(pairs.count == 3)
        #expect(Set(pairs.map(\.row)).count == 3)
        #expect(Set(pairs.map(\.col)).count == 3)
        // Brute force the optimum over all 4·3·2 injections of 3 rows into 4 columns.
        var best = Float.greatestFiniteMagnitude
        for a in 0..<4 {
            for b in 0..<4 where b != a {
                for c in 0..<4 where c != a && c != b {
                    best = min(best, cost[0][a] + cost[1][b] + cost[2][c])
                }
            }
        }
        #expect(pairs.reduce(Float(0)) { $0 + cost[$1.row][$1.col] } == best)
    }
}

struct KitTrackerTests {
    private func box(_ x: Double, _ y: Double, _ size: Double = 0.1) -> CGRect {
        CGRect(x: x, y: y, width: size, height: size)
    }

    private func detection(
        _ rect: CGRect, score: Float = 0.9, classID: Int = 1, label: String = "person"
    ) -> Detection {
        Detection(classID: classID, label: label, score: score, box: rect)
    }

    /// Feeds frames at a fixed interval and returns the tracks after the last one.
    @discardableResult
    private func run(
        _ tracker: KitTracker, _ frames: [[Detection]], step: TimeInterval = 1.0 / 15
    ) -> [Track] {
        var tracks: [Track] = []
        for (i, frame) in frames.enumerated() {
            tracks = tracker.update(frame, at: Double(i) * step)
        }
        return tracks
    }

    @Test func anObjectKeepsOneIdWhileItMoves() {
        let tracker = KitTracker()
        let frames = (0..<10).map { i in [detection(box(0.1 + Double(i) * 0.02, 0.5))] }
        let tracks = run(tracker, frames)

        #expect(tracks.count == 1)
        #expect(tracks[0].isConfirmed)
        #expect(tracks[0].hits == 10)
        #expect(tracks[0].id == 1)
    }

    @Test func aCandidateIsNotConfirmedUntilItPersists() {
        let tracker = KitTracker(options: .init(minHits: 3))
        // One frame of a detection that never comes back — a detector false positive.
        _ = tracker.update([detection(box(0.1, 0.1))], at: 0)
        #expect(tracker.current.first?.isConfirmed == false)
        let after = tracker.update([], at: 1.0 / 15)
        #expect(after.isEmpty, "an unconfirmed track must not survive a miss")
    }

    /// The frame everyone looks at. Two objects approach, cross, and separate. Greedy
    /// matching swaps their identities at the crossing; the point of an optimal assignment is
    /// that it does not.
    @Test func identitiesSurviveACrossing() {
        let tracker = KitTracker(options: .init(minHits: 2))
        var frames: [[Detection]] = []
        for i in 0..<12 {
            let t = Double(i) * 0.04
            frames.append([
                detection(box(0.20 + t, 0.5)),  // moving right
                detection(box(0.68 - t, 0.5)),  // moving left
            ])
        }
        let tracks = run(tracker, frames).sorted { $0.box.minX < $1.box.minX }

        #expect(tracks.count == 2)
        // The one that started on the left has ended up on the right, and vice versa. If the
        // ids had been swapped at the crossing, the leftmost track would be id 1 again.
        #expect(tracks.first?.id == 2)
        #expect(tracks.last?.id == 1)
    }

    @Test func aTrackCoastsThroughAnOcclusionAndKeepsItsId() {
        let tracker = KitTracker(options: .init(maxAge: 10, minHits: 2))
        // Moving right, then hidden for four frames, then back where it would have been.
        var frames: [[Detection]] = (0..<5).map { [detection(box(0.10 + Double($0) * 0.03, 0.5))] }
        frames += Array(repeating: [], count: 4)
        frames += [[detection(box(0.10 + 8 * 0.03, 0.5))]]
        let tracks = run(tracker, frames)

        #expect(tracks.count == 1)
        #expect(tracks[0].id == 1, "reappearing after an occlusion must not mint a new id")
        #expect(tracks[0].timeSinceUpdate == 0)
    }

    @Test func aTrackDiesAfterMaxAge() {
        let tracker = KitTracker(options: .init(maxAge: 3, minHits: 2))
        var frames: [[Detection]] = (0..<4).map { _ in [detection(box(0.4, 0.4))] }
        frames += Array(repeating: [], count: 5)
        #expect(run(tracker, frames).isEmpty)
    }

    /// A partly-occluded object is still detected, just faintly. Dropping everything under
    /// the display threshold drops exactly the frames where identity is hard to keep.
    @Test func aFaintDetectionKeepsAnEstablishedTrackAlive() {
        let tracker = KitTracker(options: .init(highScore: 0.5, lowScore: 0.1, minHits: 2))
        var frames: [[Detection]] = (0..<4).map { _ in [detection(box(0.4, 0.4))] }
        frames.append([detection(box(0.4, 0.4), score: 0.2)])  // faint
        let tracks = run(tracker, frames)

        #expect(tracks.count == 1)
        #expect(tracks[0].id == 1)
        #expect(tracks[0].timeSinceUpdate == 0, "the faint frame should have matched")
    }

    @Test func aFaintDetectionCannotCreateATrack() {
        let tracker = KitTracker(options: .init(highScore: 0.5, lowScore: 0.1, minHits: 1))
        let tracks = run(tracker, [[detection(box(0.4, 0.4), score: 0.2)]])
        #expect(tracks.isEmpty, "a faint blob on its own is not an object")
    }

    @Test func classesDoNotBleedIntoEachOther() {
        let tracker = KitTracker(options: .init(minHits: 1))
        _ = tracker.update([detection(box(0.4, 0.4), classID: 1, label: "person")], at: 0)
        // Same place, different class: a second object, not the first one relabelled.
        let tracks = tracker.update(
            [detection(box(0.4, 0.4), classID: 17, label: "cat")], at: 1.0 / 15)
        #expect(tracks.count == 2)
        #expect(Set(tracks.map(\.classID)) == [1, 17])
    }

    @Test func idsAreNeverReused() {
        let tracker = KitTracker(options: .init(maxAge: 1, minHits: 1))
        _ = tracker.update([detection(box(0.1, 0.1))], at: 0)
        _ = tracker.update([], at: 1)
        _ = tracker.update([], at: 2)
        let tracks = tracker.update([detection(box(0.1, 0.1))], at: 3)
        #expect(tracks.count == 1)
        #expect(tracks[0].id == 2, "a new object must not inherit a dead object's id")
    }

    /// One id through the whole sequence, at any frame rate this pipeline can produce.
    ///
    /// This is the test that found the real bug. IoU between consecutive frames falls with
    /// the interval — a 0.1-wide box moving 0.3 units/s overlaps 0.82 at 30 fps and 0.25 at
    /// 5 fps — so an IoU-only tracker minted a fresh id every frame at 5 fps, and 5 fps is
    /// exactly what this package's own thermal governor produces on a hot phone. It worked on
    /// the desk and would have failed in the user's hand.
    private func trackThrough(step: TimeInterval) -> [Track] {
        let tracker = KitTracker(options: .init(maxAge: 10, minHits: 2))
        for i in 0..<4 {
            // 0.3 normalized units per second, whatever the frame rate.
            _ = tracker.update(
                [detection(box(0.1 + 0.3 * Double(i) * step, 0.5))], at: Double(i) * step)
        }
        return tracker.update([], at: 4 * step)  // one missed frame
    }

    @Test func identityHoldsAtEveryFrameRateTheGovernorProduces() {
        for fps in [30.0, 15.0, 8.0, 5.0] {
            let tracks = trackThrough(step: 1 / fps)
            #expect(tracks.count == 1, "at \(fps) fps the object split into \(tracks.count)")
            #expect(tracks.first?.id == 1, "at \(fps) fps the id changed")
            #expect(tracks.first?.hits == 4, "at \(fps) fps only some frames matched")
        }
    }

    /// The live pipeline drops frames, so the gap between calls is not constant. A tracker
    /// that counts frames instead of seconds mispredicts by exactly that ratio.
    @Test func predictionUsesElapsedTimeNotFrameCount() {
        let fast = trackThrough(step: 1.0 / 30)[0].box
        let slow = trackThrough(step: 1.0 / 5)[0].box
        // Same physical speed, six times the interval — the slow one has travelled further.
        #expect(slow.minX > fast.minX)
        let expected = 0.1 + 0.3 * 4 * (1.0 / 5)
        #expect(abs(slow.minX - expected) < 0.05, "got \(slow.minX), expected ~\(expected)")
    }

    @Test func iouIsTheStandardDefinition() {
        let a = CGRect(x: 0, y: 0, width: 2, height: 2)
        #expect(KitTracker.iou(a, a) == 1)
        #expect(KitTracker.iou(a, CGRect(x: 5, y: 5, width: 1, height: 1)) == 0)
        // Half-overlap: intersection 2, union 6.
        let b = CGRect(x: 1, y: 0, width: 2, height: 2)
        #expect(abs(KitTracker.iou(a, b) - 2.0 / 6.0) < 1e-6)
        // Touching edges are not overlapping.
        #expect(KitTracker.iou(a, CGRect(x: 2, y: 0, width: 2, height: 2)) == 0)
    }

    @Test func resetForgetsEverythingButTheIdCounter() {
        let tracker = KitTracker(options: .init(minHits: 1))
        _ = tracker.update([detection(box(0.1, 0.1))], at: 0)
        tracker.reset()
        #expect(tracker.current.isEmpty)
        let tracks = tracker.update([detection(box(0.1, 0.1))], at: 0)
        #expect(tracks[0].id == 2, "reusing an id across a reset merges two objects in a chart")
    }
}
