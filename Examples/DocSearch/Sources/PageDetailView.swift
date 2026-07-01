import SwiftUI

// Shows a page large with a single rectangle over the tile that best matches the current query
// (reliable visual grounding: tiles are spatially-disjoint encoder inputs).
struct PageDetailView: View {
    let page: DocSearchModel.Page
    let model: DocSearchModel

    @State private var rect: CGRect?
    @State private var showHighlight = true

    private var aspect: CGFloat {
        max(page.image.size.width, 1) / max(page.image.size.height, 1)
    }

    var body: some View {
        VStack(spacing: 12) {
            if !model.currentQuery.isEmpty {
                Text("“\(model.currentQuery)”")
                    .font(.subheadline).foregroundStyle(.secondary)
            }

            ZStack {
                Image(uiImage: page.image)
                    .resizable()
                if showHighlight, let rect {
                    TileHighlight(rect: rect)
                }
            }
            .aspectRatio(aspect, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(.quaternary))

            if rect != nil {
                Toggle("Highlight match", isOn: $showHighlight)
                    .toggleStyle(.button).font(.callout)
            } else if !model.currentQuery.isEmpty {
                Text("No match region on this page").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding()
        .navigationTitle(page.title)
        .navigationBarTitleDisplayMode(.inline)
        .task { rect = model.bestTileRect(for: page) }
    }
}

// A glowing amber rectangle over a normalized (0...1) page region.
struct TileHighlight: View {
    let rect: CGRect

    var body: some View {
        GeometryReader { geo in
            let w = rect.width * geo.size.width
            let h = rect.height * geo.size.height
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(red: 1.0, green: 0.82, blue: 0.12).opacity(0.22))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(Color(red: 1.0, green: 0.68, blue: 0.0), lineWidth: 3))
                .frame(width: w, height: h)
                .position(x: (rect.midX) * geo.size.width, y: (rect.midY) * geo.size.height)
                .shadow(color: .orange.opacity(0.6), radius: 8)
        }
        .allowsHitTesting(false)
    }
}
