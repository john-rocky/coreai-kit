// ContentView — the ask-anything coach. The real demo is Siri speaking your own model's answer
// (recorded as OS UI), so this screen's job is to (1) say what this is in one line, (2) let you ask
// the on-device model directly (the in-app box — this is also the G1 proof: model loads + answers on
// device without Siri), and (3) teach the exact phrase to say to Siri. Asking in the box warms the
// shared model for the live voice call.

import SwiftUI

struct ContentView: View {
    @State private var status = ModelStatus.shared

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    IdentityHeader(
                        title: "Ask Gemma on your iPhone",
                        subtitle:
                            "\(Demo.modelName) — Google’s QAT int4, running entirely on device",
                        status: status)

                    SectionLabel("Ask the on-device model")
                    AskBox(modelReady: status.isReady)

                    SectionLabel("The demo: ask Siri")
                    SiriPhraseCard()

                    RecordHintCard()
                }
                .padding()
                .frame(maxWidth: 640)
                .frame(maxWidth: .infinity)
            }
            .navigationTitle(Demo.appName)
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
        }
        .task {
            status.prepare()   // load + warm once → fast in-app + Siri answers afterward
        }
    }
}

// MARK: - Record hint (the demo is Siri's OS UI, not this app)

struct RecordHintCard: View {
    var body: some View {
        DemoCard(accent: .secondary) {
            VStack(alignment: .leading, spacing: 8) {
                Label("This screen is the coach — the demo is Siri", systemImage: "record.circle")
                    .font(.subheadline.weight(.semibold))
                Text("Record the OS as Siri speaks your model’s answer. The first answer after launch "
                    + "is slower (the model loads once, ~4.4 GB); ask once in the box above to warm "
                    + "it, then keep this app recent so the model stays resident for Siri.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
