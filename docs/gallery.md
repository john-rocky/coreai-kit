# See it running

**[Watch the 0.4.1 Mac demo](https://github.com/john-rocky/coreai-assets/blob/main/kit/coreaikit-0.4.1-mac.mp4):
Qwen3 0.6B chat → VoxCPM 0.5B speech.** Recorded on a Mac Studio M4 Max with the
public exact `0.4.1` examples and Xcode 27 beta 5 (`27A5237l`). Builds and model downloads
are omitted; the video shows cached runs. The [reproduction record](https://github.com/john-rocky/coreai-assets/blob/main/kit/coreaikit-0.4.1-mac.md)
includes commands, model pins, and recording details. The separate
[release evidence](https://github.com/john-rocky/coreai-kit/releases/tag/0.4.1)
includes the empty-cache first download, two-turn chat, and Japanese streaming checks.

Earlier real-device captures (July 2026; iPhone 17 Pro / M4 Max). These illustrate
other capabilities and are not 0.4.1 validation evidence. Captions lead
with the one-line call where a task op covers it; each cell links to the kit example —
or zoo app — that runs the same model. (Media lives in
[coreai-assets](https://github.com/john-rocky/coreai-assets), so cloning this repo stays fast.)

| | | |
|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/chat-youtu.gif" alt="On-device chat" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/transcribe-whisper.gif" alt="Speech-to-text" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/diarize-sortformer.gif" alt="Speaker diarization" width="250"> |
| `ChatSession` — chat, Youtu-LLM-2B<br>[`ChatDemo`](../Examples/ChatDemo) | `CoreAI.transcribe` — Whisper v3 turbo<br>[`Transcribe`](../Examples/Transcribe) | `CoreAI.transcribeMeeting` — Sortformer + Parakeet<br>[`Meeting`](../Examples/Meeting) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/vlm-holo2.gif" alt="Computer-use VLM" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/agent-fastcontext.gif" alt="Repo-exploration agent" width="250"> |  |
| `KitVisionModel` — screen VLM, Holo2-4B<br>[`VLChat`](../Examples/VLChat) | Repo agent — FastContext-4B<br>[zoo `CoreAIChat`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIChat) | `CoreAI.detect` — RF-DETR nano, no NMS<br>[`DetectCamera`](../Examples/DetectCamera) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/seg-sam3.jpg" alt="Promptable segmentation" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/depth-da3.jpg" alt="Depth estimation" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/sr-adcsr.jpg" alt="Super-resolution" width="250"> |
| Segmentation — SAM 3<br>[zoo](https://github.com/john-rocky/coreai-model-zoo) | `CoreAI.estimateDepth` — Depth Anything 3<br>[`DepthCamera`](../Examples/DepthCamera) | `CoreAI.upscale` — AdcSR ×4<br>[`UpscaleDemo`](../Examples/UpscaleDemo) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/pii-gliner2.jpg" alt="PII redaction" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/ocr-glm.jpg" alt="Document OCR" width="250"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/bitcpm.gif" alt="Ternary LLM" width="250"> |
| `CoreAI.redact` — PII, GLiNER2<br>[`InfoExtract`](../Examples/InfoExtract) | `CoreAI.read` — GLM-OCR 0.9B, ~4 s/page<br>[`ReadDoc`](../Examples/ReadDoc) | 1.58-bit ternary — BitCPM-8B in ~2.1 GB<br>[zoo `CoreAIChat`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIChat) |

| | |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/imagegen-glm.jpg" alt="Text-to-image" width="390"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/edit-flux.jpg" alt="In-context image editing" width="390"> |
| Text→image — GLM-Image<br>[zoo `CoreAIImageGen`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIImageGen) | In-context edit — FLUX.2 klein<br>[zoo `CoreAIImageGen`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIImageGen) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/t2v-ltx.gif" alt="Text-to-video" width="390"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/splat-tripo.gif" alt="Photo to 3D gaussian splat" width="390"> |
| Text→video — LTX-Video 2B<br>[zoo `CoreAIVideo`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIVideo) | Photo→3D splat — TripoSplat<br>[zoo `TripoSplatMac`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/TripoSplatMac) |
| <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/dllm-llada.gif" alt="Diffusion LLM" width="390"> | <img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/readdoc-mineru.gif" alt="Document parsing" width="390"> |
| Diffusion LLM (parallel denoise) — LLaDA-8B<br>[`DiffuseChat`](../Examples/DiffuseChat) | `CoreAI.read` — MinerU2.5, doc→Markdown<br>[`ReadDoc`](../Examples/ReadDoc) |

<p align="center"><img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/forecast-timesfm.jpg" alt="Time-series forecasting" width="720"><br><code>CoreAI.forecast</code> — TimesFM 2.5, ~25 ms/forecast on iPhone · <a href="../Examples/Forecast"><code>Forecast</code></a></p>

<p align="center"><img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/kit/coder-ornith.gif" alt="Agentic coding on Mac" width="720"><br>Agentic coding — Ornith-1.0-9B on M4 Max · <a href="https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIChatMac">zoo <code>CoreAIChatMac</code></a></p>
