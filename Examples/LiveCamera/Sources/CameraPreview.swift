// CameraPreview.swift — the live feed, drawn by the compositor.
//
// `AVCaptureVideoPreviewLayer` renders the session directly, so showing the camera costs
// no per-frame CPU and competes with nothing: the pipeline pays only for inference. This
// is why `LiveWatch` hands out its `captureSession` at all.

import AVFoundation
import CoreAIOps
import SwiftUI

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession?

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.videoGravity = .resizeAspectFill
        view.previewLayer.session = session
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {
        if view.previewLayer.session !== session { view.previewLayer.session = session }
    }
}

/// Normalized boxes over the preview.
///
/// **TODO — the boxes are offset, and this view is why.** It maps normalized coordinates
/// straight onto its own bounds, which is only correct when the preview and the frame the
/// model was fed share an aspect ratio and nothing is cropped. Neither holds: the preview
/// runs at the session preset (16:9) under `.resizeAspectFill`, the model is fed 3:4. Seen
/// on device 2026-08-03.
///
/// The correction needs the frame size, which `LiveResult` does not carry yet (see its
/// `TODO`). `Examples/DetectCamera`'s `DetectionOverlay` has the working version to lift:
/// `scale = max(view.w / frame.w, view.h / frame.h)` with centre offsets, one `ZStack`
/// clipped once at the overlay bounds.
struct DetectionOverlay: View {
    let detections: [Detection]

    var body: some View {
        GeometryReader { geometry in
            ForEach(detections) { detection in
                let box = CGRect(
                    x: detection.box.origin.x * geometry.size.width,
                    y: detection.box.origin.y * geometry.size.height,
                    width: detection.box.width * geometry.size.width,
                    height: detection.box.height * geometry.size.height)
                Rectangle()
                    .stroke(.green, lineWidth: 2)
                    .frame(width: box.width, height: box.height)
                    .overlay(alignment: .topLeading) {
                        Text("\(detection.label) \(Int(detection.score * 100))%")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 4)
                            .background(.green)
                            .foregroundStyle(.black)
                            .offset(y: -14)
                    }
                    .position(x: box.midX, y: box.midY)
            }
        }
        .allowsHitTesting(false)
    }
}

/// What the pipeline is achieving, as opposed to what it was asked for. The requested frame
/// rate is not an interesting number on a phone; this is.
struct StatsBadge: View {
    let stats: LiveStats?

    var body: some View {
        if let stats {
            HStack(spacing: 10) {
                label("\(Int(stats.framesPerSecond.rounded())) fps")
                label(String(format: "%.0f ms", stats.latency.seconds * 1000))
                if stats.dropped > 0 { label("−\(stats.dropped)") }
                if stats.thermalState != .nominal {
                    label(thermalName)
                        .foregroundStyle(.orange)
                }
            }
            .font(.caption.monospaced())
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.55), in: Capsule())
            .foregroundStyle(.white)
        }
    }

    private func label(_ text: String) -> some View { Text(text) }

    /// Shown only when it is not nominal — a governed pipeline is running below its target
    /// rate on purpose, and a readout that does not say so looks like a bug.
    private var thermalName: String {
        switch stats?.thermalState {
        case .fair: "fair"
        case .serious: "hot · governed"
        case .critical: "critical · governed"
        default: ""
        }
    }
}
