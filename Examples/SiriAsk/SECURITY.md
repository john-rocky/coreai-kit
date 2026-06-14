# Gemma (SiriAsk) — agentic security note (WWDC26 347)

The ask-anything rework makes the security story almost trivial: **two of the three Lethal Trifecta
legs are absent by construction**, and the third (act/communicate) was already absent. There is no
retrieval, no untrusted data channel, no tool, and no destructive intent.

## Lethal Trifecta analysis

| Leg | Present? | Why |
|---|---|---|
| Access to private data | **No** | the model is given only the user's own typed/spoken question — no notes, files, index, or other app data are read into the prompt |
| Exposure to untrusted content | **No** | there is no retrieval and no clipped/shared content; the only input is the first-party question |
| Ability to act / communicate externally | **No** | the answering session is **tool-less** and there is no side-effecting App Intent (the earlier delete/summarize/open intents were removed) |

With no untrusted input and no capability to actuate, there is nothing for a prompt injection to
hijack — the attack the notes-RAG version had to defend against cannot arise here.

## Checklist (347)

- [x] **Prompt data sources listed + trust-marked.** Two, both trusted: the system instruction
  (`ModelHost.askInstructions`) and the user's question (in-app `TextField` or Siri's
  `requestValueDialog`). No untrusted source enters the prompt.
- [x] **Every tool/intent classified by worst-case side effect.** One intent, `AskGemmaIntent`,
  read-only (`authenticationPolicy` default; `openAppWhenRun = false`). No financial / exfil /
  data-loss / navigation intent exists in the app.
- [x] **Stored-injection surface.** Empty: the model's output is only spoken/displayed (transient
  `IntentDialog` / UI). No tool writes a model string anywhere it is later re-read.
- [x] **Side-effecting actions confirmed / never on a locked device.** N/A — there is no
  side-effecting action. Read-only ask is intentionally lock-screen-friendly.
- [x] **No "donate everything."** The app makes no `IntentDonationManager` donations.

## Residual risk (honest)

- **Model-output safety is out of 347's scope and this app's.** Whether Gemma's *answer itself* is
  harmful depends on the chosen model's own guardrails — ask-anything surfaces the base model's
  behavior directly (no app-side grounding constrains it). That is a model-choice consideration, not
  an agentic-security one.
- **On-device load + the Siri voice flow are device-verified by the user** (G1/G2).
