import CoreAIOps
import SwiftUI
import WebKit

struct FormView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = FormModel()

    var body: some View {
        VStack(spacing: 10) {
            ScreenHeader(
                title: "Form",
                subtitle: "Copy anywhere — the matching field of the form fills itself; a secret is refused")
            HStack {
                Toggle("Watch", isOn: Binding(
                    get: { model.watching },
                    set: { model.setWatching($0, runtime: runtime) }))
                    .toggleStyle(.switch)
                    .disabled(!runtime.isReady)
                Button("Clear form") { model.reset() }
                Spacer()
                Text("\(model.filledCount) of \(FormModel.fields.count) fields")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Copy from here (or from any app)").font(.caption).foregroundStyle(.secondary)
                    ScrollView {
                        Text(highlighted(model.source, model.highlight))
                            .font(.callout)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                    }
                    .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.4)))
                }
                .frame(width: 320)
                VStack(alignment: .leading, spacing: 4) {
                    Text("The form").font(.caption).foregroundStyle(.secondary)
                    FormWebView(values: model.values, version: model.fillVersion)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .frame(maxWidth: .infinity)
            }
            .frame(minHeight: 440)
            Text(model.status).font(.callout).foregroundStyle(.secondary)
            List(model.events) { event in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: event.refused ? "hand.raised.fill" : (event.field != nil ? "arrow.right.circle.fill" : "minus.circle"))
                        .foregroundStyle(event.refused ? .red : (event.field != nil ? .green : .secondary))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(event.line).font(.callout.bold())
                        Text(event.text.replacingOccurrences(of: "\n", with: " ")).font(.caption).lineLimit(1)
                        Text("\(ms(event.milliseconds)) · \(event.confidence.formatted(.number.precision(.fractionLength(2))))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .frame(minHeight: 90)
        }
        .padding()
        .task {
            await autoplay.run(.form, runtime: runtime) {
                model.reset()
                model.setWatching(true, runtime: runtime)
            }
        }
    }

    private func highlighted(_ source: String, _ range: Range<String.Index>?) -> AttributedString {
        var text = AttributedString(source)
        if let range, let lower = AttributedString.Index(range.lowerBound, within: text),
            let upper = AttributedString.Index(range.upperBound, within: text) {
            text[lower..<upper].backgroundColor = .yellow.opacity(0.45)
        }
        return text
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
