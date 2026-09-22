// ContextModel — screen: an agent's transcript with its tool results, compressed by relevance
// to the question being answered. Each tool result is prefilled once and asked one score —
// how much of it bears on the user's question — and the ones below the line are dropped
// from the context, their token count with them. The most-viewed System One post of the
// week did exactly this to make context compaction instant; here the model that scores is
// on the machine, and nothing is summarised or rewritten.
//
// Tokens are counted by the model's own tokenizer: a tool result's tokens are its prefill
// length minus the empty prefix's. On the sample (2026-09-23, MiniCPM5 2B) every result the
// question depends on scores 0.94 or above and every unrelated one 0.62 or below; the line
// sits at 0.8.

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class ContextModel {
    struct Item: Identifiable {
        let id = UUID()
        let tool: String
        let text: String
        var tokens = 0
        var answer: Decision.Answer?
        var relevance: Double? { answer?.score }
        func kept(at threshold: Double) -> Bool? { relevance.map { $0 >= threshold } }
    }

    nonisolated static let levels = ["none of it", "some of it", "most of it"]
    /// Expected level at or above which a tool result stays in the context.
    var threshold = 0.8

    var question = ContextModel.sampleQuestion
    var items: [Item] = ContextModel.sampleItems
    var status = "Compress — every tool result gets one decision, the unrelated ones drop out."
    var working = false
    private(set) var decided = false

    var totalTokens: Int { items.map(\.tokens).reduce(0, +) }
    var keptItems: [Item] { items.filter { $0.kept(at: threshold) == true } }
    var keptTokens: Int { keptItems.map(\.tokens).reduce(0, +) }
    var totalMilliseconds: Double { items.compactMap { $0.answer?.timing.milliseconds }.reduce(0, +) }
    /// One line of every score, for the hands-off log.
    var detail: String { items.map { "\($0.tool.prefix(22))=\($0.relevance.map { $0.formatted(.number.precision(.fractionLength(2))) } ?? "-")/\($0.tokens)" }.joined(separator: " | ") }

    var relevanceQuestion: Decision.Question {
        .score("The user asked: \"\(question)\" How much of this tool result is relevant to answering that?", levels: Self.levels)
    }

    func loadSample() {
        question = Self.sampleQuestion
        items = Self.sampleItems
        decided = false
        status = "Sample transcript loaded — Compress."
    }

    func run(_ runtime: DecideRuntime) {
        guard !working, !items.isEmpty else { return }
        working = true
        decided = false
        for index in items.indices { items[index].answer = nil }
        let question = relevanceQuestion
        Task {
            defer { working = false }
            do {
                let decider = try await runtime.ready()
                let overhead = try await decider.prefill("").tokens
                for index in items.indices {
                    status = "Scoring \(items[index].tool)…"
                    let prefilled = try await decider.prefill(items[index].text)
                    items[index].tokens = max(0, prefilled.tokens - overhead)
                    items[index].answer = try await prefilled.decide(question)
                }
                decided = true
                status = "\(items.count) tool results, \(totalTokens) tokens → kept \(keptItems.count), \(keptTokens) tokens · \(items.count) decisions in \(ms(totalMilliseconds))"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - Sample

    static let sampleQuestion = "why does the login test fail after the auth refactor?"

    /// Thirteen tool results an agent collected during a session; four bear on the question.
    static let sampleItems: [Item] = [
        Item(tool: "ls", text: """
            Package.swift
            Package.resolved
            README.md
            CHANGELOG.md
            LICENSE
            Sources/
              Auth/
                Session.swift
                Credentials.swift
                Keychain.swift
              Widgets/
                WidgetClient.swift
                WidgetModels.swift
                Pagination.swift
              Support/
                Logging.swift
                Retry.swift
            Tests/
              LoginTests.swift
              WidgetClientTests.swift
              PaginationTests.swift
            docs/
              getting-started.md
              api.md
            scripts/
              lint.sh
              release.sh
            """),
        Item(tool: "read README.md", text: """
            # WidgetKit client

            A Swift package for talking to the Widget API from iOS and macOS apps. It handles \
            pagination, retries with backoff, and structured logging, and ships with a small \
            command-line tool for trying requests.

            ## Installation

            Add the package in Xcode (File → Add Package Dependencies) or in Package.swift:

                .package(url: "https://example.com/widgetkit", from: "1.4.0")

            ## Usage

                let client = WidgetClient(token: token)
                let page = try await client.widgets(page: 1)
                for widget in page.items { print(widget.name) }

            ## Contributing

            Run `scripts/lint.sh` before opening a pull request. Tests run with `swift test`.
            """),
        Item(tool: "git log -3", text: """
            a41c9e2 refactor auth into Session (2 days ago)
            7b30d15 update README badges (3 days ago)
            1f8e0c4 bump version to 1.4.2 (5 days ago)
            """),
        Item(tool: "read Sources/Auth/Session.swift", text: """
            import Foundation

            /// A signed-in session: the access token and when it stops being valid.
            public struct Session: Sendable {
                public let accessToken: String
                public let refreshToken: String
                public let issuedAt: Date
                /// Seconds the access token is valid for. Lowered in the refactor so that a
                /// leaked token is useful for less time.
                public static var lifetime: TimeInterval = 60   // was 3600

                public var expiresAt: Date { issuedAt.addingTimeInterval(Self.lifetime) }
                public var isExpired: Bool { Date() >= expiresAt }

                /// Refreshes only when the token has expired; a fresh token is returned as is.
                public func refreshed(using client: AuthClient) async throws -> Session {
                    guard isExpired else { return self }
                    return try await client.refresh(refreshToken)
                }
            }
            """),
        Item(tool: "swift test", text: """
            Test Suite 'All tests' started
            Test Case 'LoginTests.testSignIn' passed (0.412 seconds)
            Test Case 'LoginTests.testRefreshKeepsSession' failed
              XCTAssertEqual failed: ("401") is not equal to ("200")
              at LoginTests.swift:31 — the request 90 seconds after sign-in was rejected
            Test Case 'WidgetClientTests.testFirstPage' passed (0.203 seconds)
            Test Case 'PaginationTests.testCursor' passed (0.011 seconds)
            Executed 4 tests, with 1 failure
            """),
        Item(tool: "read Package.swift", text: """
            // swift-tools-version: 5.10
            import PackageDescription

            let package = Package(
                name: "WidgetKit",
                platforms: [.iOS(.v17), .macOS(.v14)],
                products: [.library(name: "WidgetKit", targets: ["WidgetKit"])],
                dependencies: [
                    .package(url: "https://github.com/apple/swift-log", from: "1.5.0"),
                    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
                ],
                targets: [
                    .target(name: "WidgetKit", dependencies: [.product(name: "Logging", package: "swift-log")]),
                    .testTarget(name: "WidgetKitTests", dependencies: ["WidgetKit"]),
                ]
            )
            """),
        Item(tool: "grep -n expiresIn", text: """
            Sources/Auth/Session.swift:11:    public static var lifetime: TimeInterval = 60   // was 3600
            Tests/LoginTests.swift:18:        // the test waits 90 seconds and expects the session to still be valid
            """),
        Item(tool: "weather (from an earlier question)", text: """
            Tokyo: 24 °C, light rain until the evening, wind 3 m/s from the south-east. Tomorrow 27 °C and clear.
            """),
        Item(tool: "read CHANGELOG.md", text: """
            ## 1.4.2
            - Auth: the access-token lifetime is now configurable (`Session.lifetime`); the default was \
            lowered for security. Callers that hold a session across long operations should refresh it.
            - Widgets: cursor pagination for the list endpoint.
            - Fixed a retry loop on 429 responses.

            ## 1.4.1
            - Logging: request ids in every log line.
            """),
        Item(tool: "list open pull requests", text: """
            #88 fix typo in docs (docs/getting-started.md) — opened 4 days ago by mira
            """),
        Item(tool: "read docs/api.md", text: """
            # API

            ## WidgetClient

            `init(token:baseURL:session:)` — a client for one account. `baseURL` defaults to the \
            production endpoint; pass a staging URL in tests.

            `widgets(page:pageSize:)` — one page of widgets, newest first. Pages are numbered from 1; \
            `pageSize` is capped at 100 by the server.

            `widgets(after cursor:)` — cursor pagination for the list endpoint (1.4.2 and later). The \
            cursor is opaque; keep it as a string.

            `widget(id:)` — one widget by id, or `WidgetError.notFound`.

            `create(_:)`, `update(_:)`, `delete(id:)` — the write calls. `update` sends only the fields \
            that changed. All three retry once on a network error and never on a 4xx.

            ## Retry

            `Retry.policy` — attempts (default 3), base delay (default 0.5 s) and the multiplier. \
            A 429 waits for the server's `Retry-After` header when present.

            ## Logging

            Every request logs one line at `.debug` with the request id, method, path and elapsed \
            milliseconds; errors log at `.error` with the response body truncated to 200 characters.
            """),
        Item(tool: "read Sources/Widgets/Pagination.swift", text: """
            import Foundation

            /// A page of results with the cursor that fetches the next one.
            public struct Page<Item: Decodable & Sendable>: Decodable, Sendable {
                public let items: [Item]
                public let nextCursor: String?
                public var hasMore: Bool { nextCursor != nil }
            }

            /// Walks every page of a cursor-paginated endpoint.
            public struct PageSequence<Item: Decodable & Sendable>: AsyncSequence {
                public typealias Element = Item
                let fetch: @Sendable (String?) async throws -> Page<Item>

                public struct AsyncIterator: AsyncIteratorProtocol {
                    var buffer: [Item] = []
                    var cursor: String?
                    var done = false
                    let fetch: @Sendable (String?) async throws -> Page<Item>

                    public mutating func next() async throws -> Item? {
                        if buffer.isEmpty, !done {
                            let page = try await fetch(cursor)
                            buffer = page.items
                            cursor = page.nextCursor
                            done = page.nextCursor == nil
                        }
                        return buffer.isEmpty ? nil : buffer.removeFirst()
                    }
                }

                public func makeAsyncIterator() -> AsyncIterator { AsyncIterator(fetch: fetch) }
            }
            """),
        Item(tool: "read scripts/release.sh", text: """
            #!/bin/bash
            # release.sh <version> — tags the release and pushes the tag.
            set -euo pipefail
            VERSION=${1:?usage: release.sh <version>}
            git diff --quiet || { echo "working tree not clean"; exit 1; }
            swift test
            sed -i '' "s/^## Unreleased/## $VERSION/" CHANGELOG.md
            git commit -am "release $VERSION"
            git tag -a "$VERSION" -m "release $VERSION"
            git push origin main --tags
            echo "released $VERSION"
            """),
    ]
}
