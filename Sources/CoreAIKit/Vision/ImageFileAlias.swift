// ImageFileAlias.swift — `ImageFile` lives in CoreAIKitVision (next to ImagePreprocessor /
// CameraFeed); this alias keeps it reachable through a bare `import CoreAIKit` so the VL
// QuickStart snippet stays one import.

import CoreAIKitVision

public typealias ImageFile = CoreAIKitVision.ImageFile
