import Photos
import SwiftUI

struct SearchView: View {
    @State private var model = SearchModel()
    @State private var query = ""

    private let columns = [GridItem(.adaptive(minimum: 110), spacing: 2)]

    var body: some View {
        VStack(spacing: 0) {
            searchBar
            statusBar
            ScrollView {
                LazyVGrid(columns: columns, spacing: 2) {
                    ForEach(model.results, id: \.localIdentifier) { asset in
                        ThumbnailView(asset: asset)
                            .aspectRatio(1, contentMode: .fill)
                            .clipped()
                    }
                }
            }
        }
        .task { await model.start() }
    }

    private var searchBar: some View {
        HStack {
            TextField("red bike at the beach…", text: $query)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .onSubmit { Task { await model.search(query) } }
            Button("Search") { Task { await model.search(query) } }
                .disabled(query.isEmpty)
        }
        .padding(10)
    }

    private var statusBar: some View {
        HStack(spacing: 12) {
            Text(model.phase.label)
            Text("\(model.indexedCount) indexed")
            if let ms = model.searchMilliseconds {
                Text(String(format: "search %.0f ms", ms))
            }
            Spacer()
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)
        .padding(.horizontal, 12)
        .padding(.bottom, 6)
    }
}

struct ThumbnailView: View {
    let asset: PHAsset
    @State private var image: UIImage?

    private static let manager = PHCachingImageManager()

    var body: some View {
        GeometryReader { geo in
            Group {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Color.gray.opacity(0.15)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
            .onAppear { load(side: geo.size.width) }
        }
    }

    private func load(side: CGFloat) {
        let scale = UIScreen.main.scale
        let target = CGSize(width: side * scale, height: side * scale)
        Self.manager.requestImage(
            for: asset, targetSize: target, contentMode: .aspectFill, options: nil
        ) { result, _ in
            if let result { image = result }
        }
    }
}
