import PhotosUI
import SwiftUI

struct DocSearchView: View {
    @State private var model = DocSearchModel()
    @State private var query = ""
    @State private var pickerItems: [PhotosPickerItem] = []

    private let suggestions = [
        "total revenue in the third quarter",
        "how many employees",
        "when is payment due",
        "price of a cappuccino",
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                statusBar
                if model.isReady { searchControls }
                resultsList
            }
            .navigationTitle("Visual Doc Search")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    PhotosPicker(selection: $pickerItems, maxSelectionCount: 8, matching: .images) {
                        Image(systemName: "plus.rectangle.on.rectangle")
                    }
                    .disabled(!model.isReady)
                }
            }
            .navigationDestination(for: DocSearchModel.Page.self) { page in
                PageDetailView(page: page, model: model)
            }
        }
        .task { await model.start() }
        .onChange(of: pickerItems) { _, items in
            guard !items.isEmpty else { return }
            Task {
                for (i, item) in items.enumerated() {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let img = UIImage(data: data) {
                        await model.addImportedImage(img, title: "Imported \(model.pages.count + 1 - items.count + i + 1)")
                    }
                }
                pickerItems = []
            }
        }
    }

    private var statusBar: some View {
        Group {
            switch model.phase {
            case .ready:
                if let ms = model.searchMilliseconds {
                    Text(String(format: "Ranked %d pages in %.0f ms · tap a result to see the match",
                                model.pages.count, ms))
                } else {
                    Text("\(model.pages.count) pages · add your own with ＋, then search")
                }
            case .error(let m):
                Text(m).foregroundStyle(.red)
            default:
                HStack(spacing: 8) { ProgressView(); Text(model.phase.label) }
            }
        }
        .font(.caption).foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal).padding(.vertical, 8)
    }

    private var searchControls: some View {
        VStack(spacing: 8) {
            HStack {
                TextField("Ask about a page…", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .onSubmit { Task { await model.search(query) } }
                Button("Search") { Task { await model.search(query) } }
                    .buttonStyle(.borderedProminent)
                    .disabled(query.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack {
                    ForEach(suggestions, id: \.self) { s in
                        Button(s) { query = s; Task { await model.search(s) } }
                            .font(.caption).buttonStyle(.bordered)
                    }
                }
            }
        }
        .padding(.horizontal)
    }

    private var resultsList: some View {
        let rows = model.results.isEmpty
            ? model.pages.map { DocSearchModel.Hit(page: $0, score: 0) }
            : model.results
        return List(rows) { hit in
            NavigationLink(value: hit.page) {
                HStack(spacing: 12) {
                    Image(uiImage: hit.page.image)
                        .resizable().aspectRatio(contentMode: .fit)
                        .frame(width: 56, height: 79)
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(hit.page.title).font(.headline)
                        if !model.results.isEmpty {
                            Text(String(format: "MaxSim %.2f", hit.score))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .listStyle(.plain)
    }
}
