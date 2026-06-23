// UpscaleView.swift — pick a photo, run SinSR ×4 on-device, compare input vs result.

import CoreAIKitVision
import PhotosUI
import SwiftUI
import UIKit

struct UpscaleView: View {
    @StateObject private var engine = UpscaleEngine()
    @State private var pickerItem: PhotosPickerItem?
    @State private var original: UIImage?
    @State private var upscaled: UIImage?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        Label("Choose a photo", systemImage: "photo.on.rectangle")
                    }
                    .buttonStyle(.borderedProminent)

                    if let f = engine.downloadFraction {
                        ProgressView(value: f) {
                            Text("Downloading model… \(Int(f * 100))%")
                        }
                    }
                    Text(engine.status)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)

                    if let original {
                        imageSection(
                            "Input — \(Int(original.size.width))×\(Int(original.size.height))",
                            original)
                        Button {
                            guard let cg = original.cgImage else { return }
                            Task {
                                if let out = await engine.upscale(cg) {
                                    upscaled = UIImage(cgImage: out)
                                }
                            }
                        } label: {
                            if engine.busy {
                                ProgressView()
                            } else {
                                Label("Upscale ×4", systemImage: "wand.and.stars")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(engine.busy)
                    }

                    if let upscaled {
                        imageSection(
                            "SinSR ×4 — \(Int(upscaled.size.width))×\(Int(upscaled.size.height))",
                            upscaled)
                    }
                }
                .padding()
            }
            .navigationTitle("SinSR Upscaler")
        }
        .onChange(of: pickerItem) { _, item in
            Task {
                guard let item,
                    let data = try? await item.loadTransferable(type: Data.self),
                    let img = UIImage(data: data)
                else { return }
                original = img
                upscaled = nil
            }
        }
    }

    @ViewBuilder private func imageSection(_ title: String, _ image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 320)
                .background(Color.black.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }
}
