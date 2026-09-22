// SystemOneWireTests.swift — the `/v1/systemone` forms without weights: the order-keeping JSON
// writes what Python's `json.dumps` writes, a request parses into the kit's questions, and an
// answer is written in the reference shape.

import Foundation
import Testing

@testable import CoreAIKit

struct OrderedJSONTests {
    /// Reference strings produced by `json.dumps(x, ensure_ascii=False)` (Python 3.14).
    @Test func dumpsMatchesPython() throws {
        let cases: [(String, String)] = [
            ("{\"subject\":\"Duplicate charge\",\"body\":\"Please refund the duplicate charge.\"}",
             "{\"subject\": \"Duplicate charge\", \"body\": \"Please refund the duplicate charge.\"}"),
            ("[\"a\",1,2.5,true,null,{\"k\":\"v/w\"}]",
             "[\"a\", 1, 2.5, true, null, {\"k\": \"v/w\"}]"),
            ("{\"z\": {\"y\": [1, {\"x\": \"é\\n\\\"q\\\"\"}]}, \"a\": 0.0, \"n\": -3, \"e\": 1e-05}",
             "{\"z\": {\"y\": [1, {\"x\": \"é\\n\\\"q\\\"\"}]}, \"a\": 0.0, \"n\": -3, \"e\": 1e-05}"),
            ("{\"text\": \"tab\\tand\\u0001ctrl\"}", "{\"text\": \"tab\\tand\\u0001ctrl\"}"),
            ("{\"pair\": \"\\ud83d\\ude42 and \\u00e9\"}", "{\"pair\": \"🙂 and é\"}"),
            ("  {  }  ", "{}"),
            ("[]", "[]"),
        ]
        for (input, expected) in cases {
            #expect(try JSONValue.parse(input).dumps() == expected)
        }
    }

    @Test func keyOrderAndDuplicatesAreKept() throws {
        let value = try JSONValue.parse("{\"b\": 1, \"a\": 2, \"b\": 3}")
        #expect(value.members?.map(\.key) == ["b", "a", "b"])
        #expect(value["b"]?.doubleValue == 1)
    }

    @Test func doublesWriteLikePython() {
        #expect(JSONValue.double(0.8812).dumps() == "0.8812")
        #expect(JSONValue.double(1).dumps() == "1.0")
        #expect(JSONValue.double(0).dumps() == "0.0")
        #expect(JSONValue.double(1.05).dumps() == "1.05")
        #expect(JSONValue.int(42).dumps() == "42")
    }

    @Test func malformedInputIsRefused() {
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "\"open", "tru", "01x", "{\"a\": 1} x"] {
            #expect(throws: JSONValue.ParseError.self) { try JSONValue.parse(bad) }
        }
    }
}

struct SystemOneWireTests {
    let example = """
        {
          "state": "Help! My payouts have been failing for 3 days.",
          "model": "jev-latest",
          "questions": {
            "is_urgent": {
              "type": "noul",
              "instructions": "Does this convey urgency?",
              "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}
            },
            "queue": {
              "type": "choice",
              "instructions": "Which queue?",
              "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": null}
            },
            "anger": {
              "type": "score",
              "instructions": "How upset is the customer?",
              "criteria": ["calm", "frustrated", "very angry"]
            }
          }
        }
        """

    @Test func theReferenceRequestParses() throws {
        let request = try SystemOne.request(from: Data(example.utf8))
        #expect(request.state == "Help! My payouts have been failing for 3 days.")
        #expect(request.model == "jev-latest")
        #expect(request.structuredState == false)
        #expect(request.questions.map(\.id) == ["is_urgent", "queue", "anger"])
        #expect(request.questions[0].question.kind == .noul(yes: "Explicitly time-sensitive", no: "No urgency expressed"))
        #expect(request.questions[1].question.kind == .choice([
            .init(id: "billing", description: "billing: invoices, payments, refunds"),
            .init(id: "technical", description: "technical: bugs, outages"),
            .init(id: "other", description: "other"),
        ]))
        #expect(request.questions[2].question.kind == .score(levels: ["calm", "frustrated", "very angry"]))
        #expect(request.questions[2].question.instructions == "How upset is the customer?")
    }

    @Test func structuredStateAndInstructionsSerializeTheReferenceWay() throws {
        let body = """
            {"state": {"subject": "Duplicate charge", "body": "Refund the duplicate."}, "questions":
             {"q": {"type": "choice", "instructions": {"task": "route"}, "criteria": ["billing", "other"]}}}
            """
        let request = try SystemOne.request(from: Data(body.utf8))
        #expect(request.structuredState == true)
        #expect(request.state == "{\"subject\": \"Duplicate charge\", \"body\": \"Refund the duplicate.\"}")
        #expect(request.questions[0].question.instructions == "{\"task\": \"route\"}")
        #expect(request.questions[0].question.kind == .choice([.init("billing"), .init("other")]))
    }

    @Test func badRequestsSayWhatIsWrong() {
        func message(_ body: String) -> String {
            do {
                _ = try SystemOne.request(from: Data(body.utf8))
                return ""
            } catch let error as SystemOne.WireError {
                return error.message
            } catch {
                return "other: \(error)"
            }
        }
        #expect(message("not json").hasPrefix("invalid JSON"))
        #expect(message("{\"questions\": {}}") == "'state' is required")
        #expect(message("{\"state\": \"s\"}") == "'questions' must be an object keyed by question id")
        #expect(message("{\"state\": \"s\", \"questions\": {}}") == "'questions' is empty")
        #expect(message("{\"state\": \"s\", \"questions\": {\"q\": {\"instructions\": \"i\"}}}") == "question 'q' has no string 'type'")
        #expect(message("{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"pick\", \"instructions\": \"i\"}}}")
            == "question 'q': unknown type 'pick' (choice | score | noul)")
        #expect(message("{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"i\", \"criteria\": [\"only\"]}}}")
            == "question 'q': a choice needs at least 2 criteria, got 1")
        let many = (1...17).map { "\"o\($0)\"" }.joined(separator: ", ")
        #expect(message("{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"i\", \"criteria\": [\(many)]}}}")
            == "question 'q': this engine lists at most 16 options, got 17")
        #expect(message("{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"score\", \"instructions\": \"i\", \"criteria\": [\"one\"]}}}")
            == "question 'q': a score needs 2–10 levels, got 1")
    }

    @Test func answersAreWrittenInTheReferenceShape() {
        let timing = Decision.Timing(promptTokens: 120, reusedTokens: 100, seconds: 0.0421)
        let choiceQuestion = Decision.Question.choice("Which queue?", ["billing", "technical", "other"])
        let choice = DecisionPrompt.answer(for: choiceQuestion, probabilities: [0.88123, 0.11877, 0.0], timing: timing)
        let scoreQuestion = Decision.Question.score("How upset?", levels: ["calm", "frustrated", "very angry"])
        let score = DecisionPrompt.answer(for: scoreQuestion, probabilities: [0.0, 0.95, 0.05], timing: timing)
        let noulQuestion = Decision.Question.noul("Urgent?")
        let noul = DecisionPrompt.answer(for: noulQuestion, probabilities: [0.05, 0.95], timing: timing)

        let response = SystemOne.response(
            model: "minicpm5-2b",
            answers: [("queue", choiceQuestion, choice), ("anger", scoreQuestion, score), ("is_urgent", noulQuestion, noul)])
        let text = response.dumps()
        #expect(text.hasPrefix("{\"model\": \"minicpm5-2b\", \"answers\": {\"queue\": {\"type\": \"choice\", \"choice\": \"billing\", "
            + "\"probabilities\": {\"billing\": 0.8812, \"technical\": 0.1188, \"other\": 0.0}, \"confidence\": "))
        #expect(text.contains("\"anger\": {\"type\": \"score\", \"score\": 1.05, \"legend\": {\"0\": \"calm\", \"1\": \"frustrated\", \"2\": \"very angry\"}, "
            + "\"probabilities\": {\"0\": 0.0, \"1\": 0.95, \"2\": 0.05}, \"confidence\": "))
        #expect(text.contains("\"is_urgent\": {\"type\": \"noul\", \"noul\": 0.95, \"confidence\": 0.95}"))
        #expect(text.hasSuffix("\"usage\": {\"input_tokens\": 360, \"output_tokens\": 3}, \"timing_ms\": 126.3}"))
        // confidence is 1 − normalised entropy, the kit's certainty
        #expect(response["answers"]?["queue"]?["confidence"]?.doubleValue == (DecisionPrompt.certainty([0.88123, 0.11877, 0.0]) * 10000).rounded() / 10000)
    }

    @Test func errorsHaveTheReferenceEnvelope() {
        #expect(SystemOne.errorValue(type: "invalid_request_error", message: "'state' is required").dumps()
            == "{\"error\": {\"type\": \"invalid_request_error\", \"message\": \"'state' is required\"}}")
    }
}
