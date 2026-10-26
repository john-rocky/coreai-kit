// CoreAIKitVision — generic graph execution + typed CV pipelines over the system CoreAI
// framework. Links only the system framework: no LLM runtime is pulled in.

@_exported import CoreAIKitCore

import Foundation

/// Core AI rejects an AOT `.aimodelc` compiled for another chip with
/// `AIModelError.incompatibleCompiledAssetArchitecture(device:asset:)`. That type is not in the
/// framework's public interface, and its `localizedDescription` reads only "The operation
/// couldn’t be completed. (CoreAIDelegates.AIModelError error 0.)", which tells the person
/// holding the device nothing. Rethrow it with both chips named; any other error passes through.
func describingLoadFailure(_ error: any Error) -> any Error {
    let text = String(reflecting: error)
    guard let range = text.range(of: "incompatibleCompiledAssetArchitecture") else { return error }
    return CoreAIKitError.unsupportedDevice(
        reason: "its files were compiled for a different chip \(text[range.upperBound...])")
}

public enum VisionError: Error, LocalizedError, Sendable {
    case functionNotFound(String)
    case statefulGraphUnsupported([String])
    case unknownInput(String)
    case shapeMismatch(input: String, expected: [Int], got: [Int])
    case dtypeMismatch(input: String, expected: String)
    case unsupportedScalarType(String)
    case missingOutput(String)
    case bundleLayout(String)
    case imageRenderFailed
    case cameraAccessDenied
    case cameraUnavailable

    public var errorDescription: String? {
        switch self {
        case .functionNotFound(let name):
            return "Graph function '\(name)' not found in model"
        case .statefulGraphUnsupported(let states):
            return "GraphModel runs stateless graphs only; model declares states \(states)"
        case .unknownInput(let name):
            return "Model has no input named '\(name)'"
        case .shapeMismatch(let input, let expected, let got):
            return "Input '\(input)' expects shape \(expected), got \(got)"
        case .dtypeMismatch(let input, let expected):
            return "Input '\(input)' expects \(expected) scalars"
        case .unsupportedScalarType(let type):
            return "Unsupported scalar type \(type)"
        case .missingOutput(let name):
            return "Output '\(name)' missing after run"
        case .bundleLayout(let message):
            return message
        case .imageRenderFailed:
            return "Failed to render the preprocessed image"
        case .cameraAccessDenied:
            return "Camera access was denied"
        case .cameraUnavailable:
            return "No usable camera was found"
        }
    }
}
