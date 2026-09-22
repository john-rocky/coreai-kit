import CoreAIOps
import SwiftUI

struct SearchView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = SearchModel()

    var body: some View {
        VStack(spacing: 12) {
            ScreenHeader(
                title: "Search",
                subtitle: "Query × passages — one decision per passage, ranked by its probability")
            TextField("Query", text: $model.query)
                .textFieldStyle(.roundedBorder)
            VStack(alignment: .leading, spacing: 4) {
                Text("Passages (one per paragraph)").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $model.passagesText)
                    .font(.callout)
                    .frame(minHeight: 100, maxHeight: 160)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            }
            HStack {
                Picker("Mode", selection: $model.mode) {
                    ForEach(SearchModel.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                Button("Rank") { model.rank(runtime) }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.working)
            }
            Text(model.status).font(.callout).foregroundStyle(.secondary)
            List(model.hits) { hit in
                VStack(alignment: .leading, spacing: 4) {
                    Text(hit.passage).font(.callout).lineLimit(3)
                    ProbabilityBar(value: hit.relevance, tint: hit.relevance >= 0.5 ? .green : .gray)
                    Text("\(hit.answer.summaryLine) · \(hit.answer.timingLine)")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 2)
            }
        }
        .padding()
        .task {
            await autoplay.run(.search, runtime: runtime) { model.rank(runtime) }
        }
    }
}
