// One `import CoreAIOps` is the whole quick path: the ops plus the model-level APIs they
// ride (ChatSession, KitDetector, TextEmbedder, …), so dropping down a layer never adds
// an import — and every op's argument and result types resolve without qualification.
@_exported import CoreAIKit
@_exported import CoreAIKitEmbeddings
@_exported import CoreAIKitVision
