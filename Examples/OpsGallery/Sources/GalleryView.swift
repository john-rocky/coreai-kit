// GalleryView.swift — the one-shot ops as cards, rendered straight from the kit's own
// registry (`CoreAI.Op.gallery`): name, `summary`, and `defaultModelID` all come from
// there, so a new op in the package appears here without app changes. The live-camera
// and video-timeline ops are filtered out — they have their own example apps.

import CoreAIOps
import SwiftUI

struct GalleryView: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 170), spacing: 12)], spacing: 12
                ) {
                    ForEach(CoreAI.Op.gallery, id: \.self) { op in
                        NavigationLink(value: op) { OpCard(op: op) }
                            .buttonStyle(.plain)
                    }
                }
                .padding()
            }
            .navigationTitle("CoreAI Ops")
            .navigationDestination(for: CoreAI.Op.self) { op in
                OpDetailView(op: op)
            }
        }
    }
}

struct OpCard: View {
    let op: CoreAI.Op

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: op.icon)
                .font(.title2)
                .foregroundStyle(.tint)
            Text("CoreAI.\(op.name)")
                .font(.headline)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(op.summary)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2, reservesSpace: true)
            Text(op.defaultModelID ?? "gliner2 (pinned)")
                .font(.caption2.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
    }
}
