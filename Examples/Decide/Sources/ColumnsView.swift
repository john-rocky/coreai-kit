import CoreAIOps
import SwiftUI
import UniformTypeIdentifiers

/// A table with typed columns the model fills: open a CSV, Fill, sort by a column, save.
struct ColumnsView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = ColumnsModel()
    @State private var importing = false
    @State private var exporting = false
    @State private var editing = false

    var body: some View {
        VStack(spacing: 10) {
            ScreenHeader(
                title: "Columns",
                subtitle: "A table of texts and the columns you ask for — every row read once, one decision per column")
            controlsLayout {
                HStack(spacing: 10) {
                    Button("Open CSV…") { importing = true }
                    Button("Sample tickets") { model.loadSample() }
                    Button(editing ? "Hide columns" : "Columns…") { editing.toggle() }
                }
                HStack(spacing: 10) {
                    if model.filled > 0 {
                        Picker("Sort by", selection: $model.sortKey) {
                            Text("file order").tag(String?.none)
                            ForEach(model.columns) { column in Text(column.title).tag(String?.some(column.key)) }
                        }
                        .pickerStyle(.menu)
                    }
                    Spacer()
                    Button("Fill") { model.run(runtime) }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.working || model.rows.isEmpty)
                    Button("Save CSV…") { exporting = true }
                        .disabled(model.working || model.filled == 0)
                }
            }
            .disabled(model.working)
            .fileImporter(isPresented: $importing, allowedContentTypes: [.commaSeparatedText, .plainText, .text]) { result in
                if case .success(let url) = result { model.load(url) }
            }
            .fileExporter(
                isPresented: $exporting, document: CSVDocument(text: model.csv()), contentType: .commaSeparatedText,
                defaultFilename: model.name.replacingOccurrences(of: ".csv", with: "") + "-columns"
            ) { _ in }
            if editing {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Columns (one per line: Title = choice: q | a | b · score: q | low | high · noul: q)")
                        .font(.caption).foregroundStyle(.secondary)
                    TextEditor(text: $model.columnsText).font(.callout.monospaced())
                        .frame(height: 90)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                }
            }
            HStack {
                Text(model.name).font(.caption.bold())
                Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                if model.working {
                    ProgressView(value: Double(model.filled), total: Double(max(1, model.rows.count))).frame(width: 120)
                }
            }
            if isPhone { phoneList } else { table }
        }
        .padding()
        .onAppear { if model.rows.isEmpty { model.loadSample() } }
        .task {
            await autoplay.run(.columns, runtime: runtime, status: { model.working ? model.status : model.status + " || " + model.detail }) {
                model.loadSample()
                model.run(runtime)
                while model.working { try? await Task.sleep(for: .milliseconds(100)) }
                try? await Task.sleep(for: .seconds(autoplay.delay))
                model.sortKey = model.columns.last?.key
            }
        }
    }

    private var table: some View {
        Table(model.sortedRows) {
            TableColumn("#") { row in Text("\(row.id)").foregroundStyle(.secondary) }.width(36)
            TableColumn("Message") { row in
                Text(row.fields.last?.1 ?? row.text).lineLimit(2)
            }
            TableColumnForEach(model.columns) { column in
                TableColumn(column.title) { row in cell(row, column) }.width(min: 90, ideal: 150)
            }
        }
    }

    private var phoneList: some View {
        List(model.sortedRows) { row in
            VStack(alignment: .leading, spacing: 4) {
                Text(row.fields.last?.1 ?? row.text).font(.callout)
                if !row.answers.isEmpty {
                    HStack(spacing: 6) {
                        ForEach(model.columns) { column in cell(row, column) }
                    }
                }
            }
            .padding(.vertical, 2)
        }
    }

    @ViewBuilder
    private func cell(_ row: ColumnsModel.Row, _ column: ColumnsModel.Column) -> some View {
        let text = model.label(row, column)
        if text.isEmpty {
            Text("")
        } else {
            Text(text).font(.caption.bold())
                .padding(.horizontal, 7).padding(.vertical, 2)
                .background(Capsule().fill(tint(row, column).opacity(0.18)))
                .foregroundStyle(tint(row, column))
                .lineLimit(1)
        }
    }

    private func tint(_ row: ColumnsModel.Row, _ column: ColumnsModel.Column) -> Color {
        guard let answer = row.answers[column.key] else { return .secondary }
        switch answer.value {
        case .choice: return .blue
        case .noul(let p): return p >= 0.5 ? .green : .secondary
        case .score(let s):
            let levels = Double(s.probabilities.count - 1)
            return s.value < levels / 3 ? .red : (s.value > levels * 2 / 3 ? .green : .orange)
        }
    }
}

/// The filled table as a CSV file for the exporter.
struct CSVDocument: FileDocument {
    static let readableContentTypes: [UTType] = [.commaSeparatedText]
    var text: String

    init(text: String) { self.text = text }

    init(configuration: ReadConfiguration) throws {
        text = String(decoding: configuration.file.regularFileContents ?? Data(), as: UTF8.self)
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}
