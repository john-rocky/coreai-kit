import CoreAIOps
import SwiftUI
import WebKit

struct FormView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = FormModel()
    @State private var showSample = false

    var body: some View {
        VStack(spacing: 10) {
            ScreenHeader(
                title: "Checkout autofill",
                subtitle: "Copy an email or a message — every field fills at once, on device")
            controlsLayout {
                HStack(spacing: 10) {
                    Button("Paste") { model.paste(runtime) }
                        .buttonStyle(.borderedProminent)
                        .disabled(!runtime.isReady || model.working)
                    Button("Sample email") { model.useSample(runtime) }
                        .disabled(!runtime.isReady || model.working)
                    Button("Clear") { model.clear() }
                }
                HStack(spacing: 10) {
                    Toggle("Watch the clipboard", isOn: Binding(
                        get: { model.watching },
                        set: { model.setWatching($0, runtime: runtime) }))
                        .toggleStyle(.switch)
                        .fixedSize()
                        .disabled(!runtime.isReady)
                    Spacer()
                    if model.decisions > 0 {
                        Text("\(model.filledCount) of \(FormModel.fields.count) fields · \(model.decisions) decisions · \(ms(model.milliseconds))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            FormWebView(values: model.values, version: model.fillVersion)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .frame(minHeight: isPhone ? 360 : 430)
            Text(model.status).font(.callout).foregroundStyle(.secondary)
            if !model.fills.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(model.fills) { fill in
                        HStack(alignment: .top, spacing: 6) {
                            Text(fill.label).font(.caption.bold()).frame(width: 110, alignment: .leading)
                            Text("← \(fill.sourceLine)").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            Spacer()
                            Text(fill.confidence.formatted(.number.precision(.fractionLength(2))))
                                .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                        }
                    }
                }
                .padding(8)
                .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.3)))
            }
            DisclosureGroup("The sample email (what “Sample email” copies)", isExpanded: $showSample) {
                Text(FormModel.sampleEmail).font(.caption).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(6)
            }
            .font(.caption)
        }
        .padding()
        .task {
            await autoplay.run(.form, runtime: runtime, status: { model.status }) {
                model.clear()
                if autoplay.feed {
                    await model.fill(from: FormModel.sampleEmail, runtime: runtime)
                } else {
                    model.setWatching(true, runtime: runtime)
                }
            }
        }
    }
}

/// The checkout page, filled one field at a time through `setField(id, value)`.
#if canImport(UIKit)
struct FormWebView: UIViewRepresentable {
    let values: [String: String]
    let version: Int

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let view = WKWebView()
        view.navigationDelegate = context.coordinator
        view.loadHTMLString(FormPage.html, baseURL: nil)
        return view
    }

    func updateUIView(_ view: WKWebView, context: Context) {
        context.coordinator.apply(values, version: version, to: view)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var loaded = false
        var pending: [String: String] = [:]
        var applied = -1

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            loaded = true
            FormPage.fill(pending, in: webView)
        }

        func apply(_ values: [String: String], version: Int, to view: WKWebView) {
            guard version != applied else { return }
            applied = version
            pending = values
            if loaded { FormPage.fill(values, in: view) }
        }
    }
}
#else
struct FormWebView: NSViewRepresentable {
    let values: [String: String]
    let version: Int

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let view = WKWebView()
        view.navigationDelegate = context.coordinator
        view.loadHTMLString(FormPage.html, baseURL: nil)
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        context.coordinator.apply(values, version: version, to: view)
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var loaded = false
        var pending: [String: String] = [:]
        var applied = -1

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            loaded = true
            FormPage.fill(pending, in: webView)
        }

        func apply(_ values: [String: String], version: Int, to view: WKWebView) {
            guard version != applied else { return }
            applied = version
            pending = values
            if loaded { FormPage.fill(values, in: view) }
        }
    }
}
#endif

enum FormPage {
    /// Sets every field: a value fills it (and flashes), a missing one clears it.
    static func fill(_ values: [String: String], in view: WKWebView) {
        let payload: [String: String] = Dictionary(
            uniqueKeysWithValues: FormModel.fields.map { ($0.id, values[$0.id] ?? "") })
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
            let json = String(data: data, encoding: .utf8) else { return }
        view.evaluateJavaScript("setFields(\(json))")
    }

    static let html = """
        <!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body { font: 15px -apple-system, system-ui, sans-serif; margin: 0; padding: 16px 18px; background: #fafafa; color: #222; }
          h1 { font-size: 17px; margin: 0 0 2px; } .sub { color: #777; font-size: 12px; margin: 0 0 14px; }
          label { display: block; font-size: 12px; color: #555; margin: 10px 0 3px; }
          input, textarea { width: 100%; box-sizing: border-box; font: inherit; padding: 7px 9px; border: 1px solid #cfcfcf; border-radius: 6px; background: #fff; transition: background .6s, border-color .6s; }
          textarea { height: 62px; resize: none; }
          .flash { background: #dff7e2; border-color: #4cc26b; transition: none; }
          .row { display: flex; gap: 10px; } .row > div { flex: 1; }
          button { margin-top: 16px; width: 100%; padding: 9px; font: inherit; border: 0; border-radius: 6px; background: #1f6feb; color: #fff; }
        </style></head><body>
        <h1>Harbor Lane Roasters · Checkout</h1><p class="sub">Shipping details</p>
        <label for="name">Full name</label><input id="name">
        <label for="address">Shipping address</label><textarea id="address"></textarea>
        <div class="row"><div><label for="phone">Phone</label><input id="phone"></div>
        <div><label for="email">Email</label><input id="email"></div></div>
        <div class="row"><div><label for="order">Order reference</label><input id="order"></div></div>
        <label for="note">Delivery note</label><input id="note">
        <button type="button">Continue to payment</button>
        <script>
          function setFields(v) {
            for (const id in v) {
              const el = document.getElementById(id); if (!el) continue;
              if (el.value !== v[id]) {
                el.value = v[id];
                if (v[id]) { el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 1200); }
              }
            }
          }
        </script></body></html>
        """
}
